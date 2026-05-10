Here is a series of documents based on the provided tutorials for Triton's new low-level extension, Gluon.
For more details, please look at the official tutorials:
* [01-intro](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py)
* [02-layouts](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/02-layouts.py)
* [03-async-copy](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/03-async-copy.py)
* [04-tma](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/04-tma.py)
* [05-wgmma](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/05-wgmma.py)
* [06-tcgen05](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/06-tcgen05.py)
* [07-persistent](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/07-persistent.py)
* [08-warp-specialization.py](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py)

***

### 1. Introduction to Gluon

Gluon is a GPU programming language built upon the same compiler infrastructure as Triton. However, it operates at a lower level, granting developers more explicit control over kernel implementation, which also comes with greater responsibility. This series of documents will guide you through GPU kernel development in Gluon, starting from fundamental concepts and advancing to sophisticated optimization techniques and features of modern GPU hardware.

While Gluon and Triton share a tile-based Single Program, Multiple Data (SPMD) programming model and a Python DSL frontend, they differ significantly in their level of abstraction. Triton abstracts away many low-level details, such as tile layouts, memory management, and data movement, leaving them to the compiler. While Triton's compiler is effective for a broad range of kernels, hand-tuned, low-level code can often achieve superior performance.

Gluon exposes these low-level details, allowing developers to finely tune kernels for maximum performance. This requires a deeper understanding of GPU architecture but unlocks the potential to surpass the performance of compiler-generated code.

#### Getting Started: Kernels and Launchers

Defining and launching a Gluon kernel is syntactically similar to Triton. You use the `@gluon.jit` decorator for the kernel and invoke it from Python with a specified grid.

Here is a simple kernel that copies a single scalar value:

```python
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)

def copy_scalar(input, output):
    # Launch a single program on a 1x1 grid.
    grid = (1,)
    copy_scalar_kernel[grid](input, output, num_warps=1)

# Example usage:
input = torch.tensor([42.0], device="cuda")
output = torch.empty_like(input)
copy_scalar(input, output)
```

#### Using `constexpr` and Autotuning

Like Triton, Gluon kernels can accept compile-time constants (`constexpr`) as arguments. This is useful for parameterizing kernels with values like block sizes. The following `memcpy` kernel processes data in 1D blocks, where each program handles one block.

```python
import triton

@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    # Clamp the range to avoid out-of-bounds access
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)

def memcpy(input, output, XBLOCK):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK),)
    memcpy_kernel[grid](input, output, xnumel, XBLOCK, num_warps=1)
```

Gluon also integrates with Triton's autotuning functionality. You can define a range of configurations for your `constexpr` arguments, and Triton will benchmark them to find the optimal settings.

```python
@triton.autotune(
    configs=[triton.Config({"XBLOCK": 2**i}, num_warps=1) for i in range(8, 14)],
    key=["xnumel"],
)
@gluon.jit
def memcpy_kernel_autotune(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    # This kernel simply calls the previously defined memcpy_kernel
    memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK)

def memcpy_autotune(input, output):
    xnumel = input.numel()
    def grid(META):
        return (triton.cdiv(xnumel, META["XBLOCK"]),)
    memcpy_kernel_autotune[grid](input, output, xnumel)
```

Running this autotuner reveals that while simple, this kernel does not fully utilize the GPU's parallelism, achieving only a fraction of the peak memory bandwidth. To improve this, we need to load and store entire tiles of data at once, which requires a deep understanding of tensor layouts.

#### Key Takeaways

*   The high-level structure of writing and launching Gluon kernels is identical to Triton.
*   Gluon uses the same tile-based SPMD programming model familiar to Triton users.
*   The primary difference lies in the device-side code, where Gluon exposes low-level hardware details for fine-grained performance tuning.

***

### 2. Tensor Layouts

In Gluon, tensors must have an explicit layout that defines how their elements are distributed across the GPU's hierarchy: thread blocks (CTAs), warps, lanes (threads), and finally, registers within each lane.

The most common layout is `gl.BlockedLayout`. It describes how a tensor is organized within a block, which is hierarchically tiled. Consider this example:

```python
layout = gl.BlockedLayout(
    size_per_thread=[2, 4],
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    order=[1, 0],
)
```

This layout defines a block shape of `[64, 16]`, calculated by the element-wise product of its components.
*   `size_per_thread=[2, 4]`: Each thread owns a `2x4` tile of data.
*   `threads_per_warp=[16, 2]`: A warp is composed of a `16x2` grid of these thread tiles.
*   `warps_per_cta=[2, 2]`: A CTA (program) is composed of a `2x2` grid of these warp tiles.
*   `order=[1, 0]`: This specifies a row-major tiling order.

The choice of layout has a dramatic impact on performance, especially for memory operations.

#### 1D Memcpy with Layouts

Let's enhance our `memcpy` kernel to load and store entire tiles by specifying a layout for the tensor indices.

```python
@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # Specify the layout for the indices tensor.
    # This layout will propagate through subsequent operations.
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel

    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)
```

Benchmarking this kernel with different layouts reveals that performance is sensitive to how data is accessed. For a 1D tensor, the `size_per_thread` (aliased as `R` in the tutorial) affects the vectorization of global memory loads (`LDG.E`) and stores (`STG.E`). On modern NVIDIA GPUs with 128-byte cache lines, an `R` value that results in 128-byte memory accesses per instruction often yields the best performance by ensuring memory coalescing. However, other hardware factors, such as instruction scheduling and register dependencies, can cause smaller or larger access sizes to be more performant in certain situations.

#### 2D Memcpy and SliceLayout

For multi-dimensional tensors, layouts are even more critical. To index a 2D tensor, we need to create 1D offset vectors for each dimension and then combine them. This requires a `gl.SliceLayout`, which is derived from a parent layout by dropping one dimension.

```python
@gluon.jit
def memcpy_2d_kernel(in_ptr, out_ptr,
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK

    # Use SliceLayout to create 1D indices from the 2D parent layout
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    # Combine 1D indices to form 2D offsets
    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)
```

The optimal layout for a 2D `memcpy` depends on which dimension of the tensor is contiguous in memory. A layout that coalesces memory accesses along the contiguous dimension will perform best. If the input and output tensors have different memory layouts (e.g., one is row-major and the other is column-major), a single layout for both loading and storing will result in poor performance.

#### Layout Conversions

The solution is to use `gl.convert_layout`. This operation explicitly converts a tensor from one layout to another. While this can be an expensive operation involving data movement between threads (often via shared memory), it is usually far cheaper than performing uncoalesced global memory accesses.

```python
@gluon.jit
def memcpy_2d_inout_kernel(in_ptr, out_ptr,
                           xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,
                           layout_in: gl.constexpr, layout_out: gl.constexpr,
                           XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    # ... (compute offsets and masks for both input and output layouts)

    # Load using the optimal layout for the input tensor
    value = gl.load(in_ptr + in_offsets, mask=mask_in)

    # Convert the tensor to the optimal layout for the output tensor
    value = gl.convert_layout(value, layout_out)

    # Store using the optimal layout for the output tensor
    gl.store(out_ptr + out_offsets, value, mask=mask_out)
```

#### Key Takeaways

*   Gluon requires explicit layout management, which is critical for performance.
*   Layouts affect global memory access patterns, inter-thread communication, and shared memory bank conflicts.
*   `BlockedLayout` is used for coalesced memory access, while `SliceLayout` helps in manipulating dimensions for indexing.
*   `gl.convert_layout` is used to transform tensors between layouts, which is necessary when input and output data have different memory organizations.

***

### 3. Async Copy in Gluon

Modern GPUs feature asynchronous instructions that allow long-running operations, like memory transfers, to overlap with computation. This technique, known as pipelining, is essential for hiding memory latency and maximizing hardware utilization.

On NVIDIA GPUs, asynchronous copies (`cp.async`) transfer data between global and shared memory. This is different from `gl.load` and `gl.store`, which move data directly between global memory and registers.

#### Basic Async Copy

Here is the 1D `memcpy` kernel rewritten to use `cp.async_copy_global_to_shared`.

```python
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

@gluon.jit
def memcpy_1d_cpasync_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    offsets = pid * XBLOCK + gl.arange(0, XBLOCK, layout=layout)
    mask = offsets < xnumel

    # Allocate shared memory with a specific layout
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    smem = gl.allocate_shared_memory(gl.float32, [XBLOCK], layout=smem_layout)

    # 1. Issue the asynchronous copy from global to shared memory
    cp.async_copy_global_to_shared(smem, in_ptr + offsets, mask=mask)
    # 2. Commit the copy to a group
    cp.commit_group()
    # 3. Wait for the group to complete
    cp.wait_group(0)

    # 4. Load from shared memory to registers and store back to global
    value = smem.load(layout)
    gl.store(out_ptr + offsets, value, mask=mask)```
Even without explicit pipelining, using `cp.async` can be significantly faster than direct loads because it utilizes a dedicated hardware path for data movement.

#### Software Pipelining

To hide the latency of memory operations, we can use software pipelining. The core idea is to prefetch the data for the *next* iteration of a loop while the *current* iteration is being processed. This requires using multiple buffers in shared memory to hold the in-flight data.

The following `elementwise_add` kernel is pipelined with a configurable number of buffers (`num_buffers`).

```python
@gluon.jit
def elementwise_add_pipelined_kernel(
        a_ptr, b_ptr, c_ptr, xnumel, ynumel,
        xstride_a, ystride_a, xstride_b, ystride_b, xstride_c, ystride_c,
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr,
        smem_layout: gl.constexpr, num_buffers: gl.constexpr,
):
    # ... (setup pointers and indices)

    # Allocate multi-buffered shared memory
    dtype: gl.constexpr = a_ptr.dtype.element_ty
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], layout=smem_layout)

    # --- Pipeline Prologue: Fill the pipeline ---
    # Issue the first `num_buffers - 1` loads
    for i in gl.static_range(num_buffers - 1):
        # ... (issue async_copy for a_smem[i] and b_smem[i])
        cp.commit_group()

    # --- Pipeline Main Loop: Steady State ---
    for yoff in range(0, ynumel - (num_buffers - 1) * YBLOCK, YBLOCK):
        # Issue load for iteration `i + num_buffers - 1`
        # ... (issue async_copy for the next block)
        cp.commit_group()

        # Wait for the data from iteration `i` to be ready
        cp.wait_group(num_buffers - 1)

        # Compute on data from iteration `i`
        a_val = a_smem.index(read_idx % num_buffers).load(layout)
        b_val = b_smem.index(read_idx % num_buffers).load(layout)
        c_val = a_val + b_val
        gl.store(...)

    # --- Pipeline Epilogue: Drain the pipeline ---
    for i in gl.static_range(num_buffers - 1):
        # Wait for remaining loads and perform the final computations
        cp.wait_group(num_buffers - 2 - i)
        # ... (compute and store)
```

Pipelining provides a modest speedup for memory-bound kernels. The optimal number of buffers (pipeline depth) depends on the relative latencies of the memory and compute operations.

#### Key Takeaways

*   Asynchronous instructions like `cp.async` enable the overlap of memory transfers and computation.
*   Async copies are managed using commit groups, allowing the program to wait for multiple operations to complete.
*   Software pipelining is a loop optimization that prefetches data for future iterations to hide memory latency.
*   Shared memory layouts are crucial for performance, as they can affect bank conflicts when multiple threads access shared memory simultaneously.

***

### 4. TMA in Gluon

On Hopper and newer NVIDIA architectures, the Tensor Memory Accelerator (TMA) provides a specialized hardware path for asynchronous N-dimensional tensor transfers between global and shared memory. TMAs offer higher performance than regular async copies by using a compact "tensor descriptor" to represent memory accesses, which reduces register pressure.

#### Using Tensor Descriptors and `mbarrier`

TMA operations are initiated via tensor descriptors, which are created on the host and passed to the kernel. These descriptors contain the tensor's shape, strides, and layout information.

Synchronization for TMA reads is handled by `mbarrier` objects, which are 64-bit counters in shared memory. A typical workflow is:
1.  Initialize an `mbarrier` with a count.
2.  Tell the `mbarrier` how many bytes to expect (`mbarrier.expect`).
3.  Issue a `tma.async_copy_global_to_shared`, passing the `mbarrier`. The TMA hardware will automatically decrement the byte counter and "arrive" on the barrier (decrementing its count) when the transfer is complete.
4.  Wait for the barrier's phase to change (`mbarrier.wait`), indicating completion.

TMA stores, like `cp.async`, are tracked with implicit commit groups.

```python
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier

@gluon.jit
def memcpy_1d_tma_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(...)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    # Initialize barrier to complete after 1 arrival
    mbarrier.init(bar, count=1)

    # Expect a certain number of bytes for the transfer
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    # Issue TMA read, which will arrive on `bar` upon completion
    tma.async_copy_global_to_shared(in_desc, [pid * XBLOCK], bar, smem)

    # Wait for the barrier to complete its current phase
    mbarrier.wait(bar, phase=0)

    # Issue TMA store from shared memory to global
    tma.async_copy_shared_to_global(out_desc, [pid * XBLOCK], smem)

    # Wait for outstanding TMA stores to complete
    tma.store_wait(pendings=0)
```

#### Memory Fencing with `fence_async_shared`

TMAs access shared memory via a separate hardware unit called the "async proxy," while standard register loads/stores from shared memory use the "generic proxy." To ensure correct ordering between these two proxies, an explicit fence is required.

*   A `fence_async_shared()` is needed **before** a TMA operation if it might conflict with a preceding generic proxy access (e.g., `smem.store(...)` followed by `tma.async_copy_shared_to_global(...)`).
*   A fence is not needed after a `mbarrier.wait()` on a TMA read, as the wait guarantees the async proxy has finished writing to shared memory.

#### Pipelining with TMA

Pipelining TMA operations follows the same principles as with `cp.async`, but requires managing `mbarrier` objects for each buffer to track completion.

```python
@gluon.jit
def elementwise_add_tma_kernel(
        a_desc, b_desc, c_desc, xnumel, ynumel,
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    # ... (setup)

    # Allocate one mbarrier per buffer
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    # --- Pipeline Main Loop ---
    for ...:
        # Issue loads for the next iteration, using the next barrier
        bar = bars.index(copy_index % num_buffers)
        mbarrier.expect(bar, ...)
        tma.async_copy_global_to_shared(a_desc, ..., bar, a_smem.index(...))
        tma.async_copy_global_to_shared(b_desc, ..., bar, b_smem.index(...))

        # Wait for the data from a previous iteration to be ready
        read_phase = read_index // num_buffers & 1
        mbarrier.wait(bars.index(read_index % num_buffers), read_phase)

        # Compute and store
        a_val = a_smem.index(...).load(layout)
        b_val = b_smem.index(...).load(layout)
        c_val = a_val + b_val
        c_smem.store(c_val)
        fence_async_shared() # Fence before TMA store
        tma.async_copy_shared_to_global(c_desc, ..., c_smem)
```
Switching from `cp.async` to TMA can yield a significant performance boost due to the dedicated hardware path and reduced register pressure. This often allows for larger block sizes, further improving performance.

#### Key Takeaways

*   TMA uses a faster, dedicated hardware path for asynchronous memory transfers on Hopper+ GPUs.
*   TMA operations are defined by `TensorDescriptor` objects and synchronized using `mbarrier`s for reads and commit groups for stores.
*   `fence_async_shared` is required to ensure memory ordering between the async proxy (TMA) and the generic proxy (register loads/stores).
*   TMAs reduce register pressure, often enabling larger tile sizes and deeper pipelines.

***

### 5. Warp-Group MMA

Hopper GPUs introduced Warp-Group MMA (WGMMA), an asynchronous instruction for performing matrix multiply-accumulate operations using Tensor Cores. Being asynchronous, WGMMA can be pipelined to overlap with memory transfers and other computations.

#### Using `warpgroup_mma`

The `warpgroup_mma` instruction performs the operation `d = a * b + c`.
*   Operand `a` can be in registers or shared memory.
*   Operand `b` must be in shared memory.
*   Operand `c` (the accumulator) must be in registers.

WGMMA has strict layout requirements. The accumulator (`c`) must have an `NVMMADistributedLayout`, and operands in shared memory must have an `NVMMASharedLayout`. The instruction shape (`[m, n, k]`) is also constrained, requiring at least 4 warps (a "warp group") to execute.

Completion is tracked similarly to TMA stores. Issuing a `warpgroup_mma` operation implicitly commits it to a group, and `warpgroup_mma_wait(num_outstanding=N)` is used to wait until N or fewer operations are in flight.

#### Blocked Matmul with WGMMA

A standard blocked matrix multiplication can be implemented by tiling the operation along the K dimension within each kernel. Each program computes one `[BLOCK_M, BLOCK_N]` tile of the output matrix.

```python
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma, warpgroup_mma_wait

@gluon.jit
def blocked_matmul_kernel(a_desc, b_desc, c_desc, TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr):
    # ... (setup offsets and shared memory buffers)

    # Initialize accumulator with a specific WGMMA layout
    mma_layout: gl.constexpr = pick_wgmma_layout(...)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    bar = gl.allocate_shared_memory(...)
    mbarrier.init(bar, count=1)

    for k in range(0, K, BLOCK_K):
        # 1. Load tiles of A and B from global to shared memory using TMA
        tma.async_copy_global_to_shared(a_desc, ...)
        tma.async_copy_global_to_shared(b_desc, ...)
        mbarrier.wait(bar, ...)

        # 2. Issue the asynchronous WGMMA
        # `a_smem` and `b_smem` are shared memory descriptors
        acc = warpgroup_mma(a_smem, b_smem, acc, is_async=True)

        # 3. Wait for the WGMMA to complete before the next iteration
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    # ... (store result from accumulator to global memory)
```

#### Pipelining WGMMA

To improve performance, the WGMMA instruction can be overlapped with the TMA loads of the next tile. This requires double-buffering the operand shared memory (`a_smem` and `b_smem`). The schedule becomes:
1.  Load data for iteration `k` into buffer 0.
2.  Wait for load `k` to complete.
3.  **Wait** for WGMMA from iteration `k-1` to complete.
4.  **Issue** WGMMA for iteration `k` using data from buffer 0.
5.  **Simultaneously**, issue load for iteration `k+1` into buffer 1.
6.  Repeat, alternating buffers.

```python
@gluon.jit
def blocked_matmul_pipelined_kernel(a_desc, b_desc, c_desc, num_warps: gl.constexpr):
    # ... (setup with double-buffered a_smem and b_smem)

    for k in range(0, K, BLOCK_K):
        # Determine current buffer index
        index = (k // BLOCK_K) % 2

        # Load next tile while computing on current tile
        tma.async_copy_global_to_shared(...) # Load into a_smem.index(index)
        tma.async_copy_global_to_shared(...) # Load into b_smem.index(index)
        mbarrier.wait(...)

        # Overlap by waiting first, then issuing the async MMA
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        acc = warpgroup_mma(a_smem.index(index), b_smem.index(index), acc, is_async=True)

    # Wait for the final WGMMA to complete
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    # ... (store result)
```
Pipelining the WGMMA provides a significant speedup, but often requires re-tuning hyperparameters like block sizes to find the new performance optimum.

#### Key Takeaways

*   WGMMA is a Hopper-specific asynchronous instruction for block-level matrix multiplication on Tensor Cores.
*   WGMMA has strict layout and shape constraints and requires a minimum of 4 warps.
*   It can be pipelined with memory operations by double-buffering operands in shared memory, leading to substantial performance gains.
*   Hyperparameter tuning (block sizes, number of warps) is crucial to achieve peak performance with WGMMA.

***

### 6. The 5th Generation Tensor Core (Blackwell)

NVIDIA's Blackwell architecture introduces the 5th generation Tensor Core, which interacts with a new memory space called **Tensor Memory (TMEM)**. TMEM is a fast, SM-local memory space that serves as an intermediary for MMA instructions.

#### Interacting with Tensor Memory (TMEM)

TMEM is a 2D memory space (128 rows x 512 columns of 32-bit cells) that is faster than shared memory but has several restrictions:
*   It is allocated in columns, with sizes being powers of two.
*   A full warp group (4 or 8 warps) is required to access it.
*   Only 2D tensors can be loaded from or stored to TMEM in Gluon.

Data in TMEM has a `TensorMemoryLayout`, and tensors in registers must be converted to a compatible layout before being stored to TMEM.

```python
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_32x32b_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
)

@gluon.jit
def tmem_example_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    # ... (load `input` from global memory to registers)

    # 1. Allocate Tensor Memory
    tmem_layout: gl.constexpr = TensorMemoryLayout(block=(64, 64), ...)
    tmem = allocate_tensor_memory(..., shape=[M, N], layout=tmem_layout)

    # 2. Get the required register layout for TMEM access
    tmem_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(...)

    # 3. Convert input tensor to the correct layout and store to TMEM
    input = gl.convert_layout(input, tmem_reg_layout)
    tmem.store(input)

    # 4. Load from TMEM back to registers
    output = tmem.load(tmem_reg_layout)

    # ... (convert back to global memory layout and store)
```

#### Blackwell MMA (`tcgen05_mma`)

The Blackwell MMA instruction, `tcgen05_mma`, is also asynchronous. Its operand requirements differ from WGMMA:
*   The **accumulator** operand must be in TMEM.
*   The LHS (`a`) operand can be in either SMEM or TMEM.
*   The RHS (`b`) operand must be in SMEM.

Completion is tracked with `mbarrier` objects. Unlike WGMMA, you must explicitly commit the MMA operation to a barrier using `tcgen05_commit(bar)` or by passing the barrier directly to the `tcgen05_mma` call.

#### Blocked Matmul with `tcgen05_mma`

A blocked matmul on Blackwell follows a similar pattern to Hopper, but with the added step of moving the accumulator to and from TMEM.

```python
@gluon.jit
def blocked_matmul_kernel(a_desc, b_desc, c_desc, ...):
    # ... (setup offsets, shared memory for A and B)

    # Allocate TMEM for the accumulator
    tmem_layout: gl.constexpr = TensorMemoryLayout(...)
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # Initialize two mbarriers: one for TMA loads, one for MMAs
    tma_bar = gl.allocate_shared_memory(...)
    mma_bar = gl.allocate_shared_memory(...)

    use_acc = False # For the first iteration, zero-initialize the accumulator
    for k in range(0, K, BLOCK_K):
        # 1. Load A and B from global to shared memory
        tma.async_copy_global_to_shared(...)
        mbarrier.wait(tma_bar, ...)

        # 2. Issue and wait on the tcgen05_mma
        tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, ...)
        use_acc = True

    # 3. Load result from TMEM to registers
    acc_reg_layout: gl.constexpr = get_tmem_32x32b_reg_layout(...)
    acc = acc_tmem.load(acc_reg_layout)

    # ... (store result to global memory)
```
A highly optimized, pipelined kernel can be constructed by double-buffering the operands in shared memory and the accumulators in tensor memory, creating a fine-grained schedule of TMA loads and `tcgen05_mma` instructions to maximize overlap.

#### Key Takeaways

*   Blackwell GPUs introduce Tensor Memory (TMEM), a fast SM-local memory space required for `tcgen05_mma` accumulators.
*   `tcgen05_mma` is an asynchronous instruction whose completion is tracked via `mbarrier`s and `tcgen05_commit`.
*   Using TMEM and `tcgen05_mma` requires careful management of tensor layouts in registers, shared memory, and tensor memory.
*   Pipelining on Blackwell involves overlapping TMA loads, `tcgen05_mma` compute, and data transfers between SMEM and TMEM.

***

### 7. Persistent Kernels

Typically, Triton kernels are launched on a large grid where each program instance handles one block of work. The GPU scheduler manages distributing these numerous, short-lived programs across the SMs. **Persistent kernels** are an alternative technique where a smaller number of long-lived programs are launched (often one per SM). Each program then loops internally to process multiple blocks of work according to a static schedule.

This approach reduces scheduler overhead and enables more advanced optimization opportunities, such as pipelining work between different output blocks, at the cost of losing the GPU's dynamic load balancing.

#### Implementing a Persistent Matmul

To convert a standard blocked matmul into a persistent one, we wrap the core logic in an outer loop that iterates through the output tiles assigned to that program. A scheduler helper class can manage the work distribution.

```python
# A simple scheduler that divides tiles linearly among programs
@aggregate
class PersistentTileScheduler:
    # ... (stores start and end tile IDs for the current program)
    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        # ... (calculates which range of tile IDs this program is responsible for)
    @gluon.jit
    def get_tile(self, idx):
        # ... (converts a local index into a global (pid_m, pid_n) tile coordinate)

@gluon.jit
def persistent_matmul_kernel(a_desc, b_desc, c_desc, ...):
    # ... (initialize MMA state, barriers, etc. once)

    # Initialize the scheduler
    scheduler = PersistentTileScheduler.initialize(...)

    # Outer loop over assigned output tiles
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        # Inner loop over K dimension (the original kernel logic)
        for k in range(0, K, BLOCK_K):
            # ... (load A and B, perform MMA)

        # ... (store result for the current tile)
```

#### Scheduling and Performance

The performance of a persistent kernel is highly sensitive to its scheduling strategy. A simple row-major scheduler can lead to poor L2 cache locality. A "grouped" or "super-grouping" scheduler, which processes tiles in a way that maximizes data reuse in the cache (e.g., processing a group of rows completely before moving to the next), can significantly improve performance.

#### Pipelining Across the Outer Loop

The main advantage of persistent kernels is the ability to pipeline operations across the boundary of output tiles. For example, the epilogue (storing the result) of tile `i` can be overlapped with the prologue (loading the first operands) of tile `i+1`.

This requires careful management of shared memory and synchronization. The liverange of the buffer used for the TMA store of tile `i` must not conflict with the operand buffers for tile `i+1`. This can be achieved by:
1.  Using more shared memory for separate buffers.
2.  "Stealing" an unused operand buffer from the next iteration for the store, if its size is sufficient.

This advanced pipelining is especially beneficial for matrices with a small K dimension, where the epilogue constitutes a larger fraction of the total execution time.

#### Key Takeaways

*   Persistent kernels use a small number of long-lived programs that internally loop over multiple blocks of work.
*   This reduces launch overhead and allows for cross-block optimizations at the cost of static scheduling.
*   The tile scheduling strategy is critical for performance, directly impacting cache locality.
*   The primary benefit is the ability to pipeline operations across the outer loop, overlapping the epilogue of one tile with the prologue of the next.

***

### 8. Warp Specialization

Warp specialization is a technique where different groups of warps within a single kernel are assigned entirely different tasks. For instance, some warps (a "producer" partition) might be dedicated solely to fetching data from memory, while others (a "consumer" partition) perform computations. This allows independent parts of a kernel to execute in parallel, relying on the GPU's warp scheduler to hide latencies more effectively than software pipelining alone.

This technique is supported on Hopper and newer GPUs in Gluon.

#### Defining a Warp-Specialized Kernel

To create a warp-specialized kernel, you define separate Python functions for each partition and use `gl.warp_specialize` to orchestrate them.
*   **Default Partition:** The main compute partition. It receives the full `num_warps` specified for the kernel.
*   **Worker Partitions:** Specialized partitions (e.g., for loading or storing). You specify the number of warps and registers for each.

Synchronization between partitions is achieved using `mbarrier`s to signal data readiness and consumption.

#### Example: Warp-Specialized Elementwise Add

An elementwise add kernel can be split into three partitions:
1.  `load_partition` (1 warp): Issues TMA loads for operands A and B into shared memory buffers. Signals `load_ready_bars` when data is available.
2.  `compute_partition` (default, N warps): Waits on `load_ready_bars`, performs the addition, stores the result to another shared memory buffer, and signals `c_ready_bars`.
3.  `store_partition` (1 warp): Waits on `c_ready_bars`, consumes the result, and issues TMA stores to global memory.

```python
@gluon.jit
def load_partition(descs, barriers, buffers, ...):
    # Loop over tiles
    # 1. Wait for a buffer to be empty (signaled by compute partition)
    # 2. Issue TMA loads into the buffer
    # 3. Loads will signal a "ready" barrier upon completion

@gluon.jit
def store_partition(descs, barriers, buffers, ...):
    # Loop over tiles
    # 1. Wait for a result buffer to be ready (signaled by compute partition)
    # 2. Issue TMA store from the buffer
    # 3. After store is in flight, signal that a previous buffer is now empty

@gluon.jit
def compute_partition(barriers, buffers, ...): # Default partition
    # Loop over tiles
    # 1. Wait for operand buffers to be ready (signaled by load partition)
    # 2. Load from SMEM, compute, signal that operand buffers are now empty
    # 3. Wait for a result buffer to be empty (signaled by store partition)
    # 4. Store result to SMEM, signal that result buffer is now ready

@gluon.jit
def elementwise_add_warp_specialized_kernel(...):
    # ... (allocate all shared memory buffers and mbarriers)

    gl.warp_specialize(
        default_args=(...),
        default_partition=compute_partition,
        worker_args=(...),
        worker_partitions=[load_partition, store_partition],
        worker_num_warps=[1, 1],
        worker_num_regs=[24, 24], # Minimal registers for memory-only warps
    )
```

#### Application to Matmul

Warp specialization is particularly powerful for complex kernels like matrix multiplication. A persistent matmul can be partitioned into:
*   **Load Partition:** Manages all TMA loads for operands A and B.
*   **MMA Partition:** Manages all asynchronous MMA instructions.
*   **Epilogue Partition (Default):** Manages loading the final accumulator from TMEM, downcasting, and storing the result to global memory.

This structure creates a deep, hardware-managed pipeline. The load partition feeds the MMA partition, which in turn feeds the epilogue partition. Each stage can be multi-buffered (e.g., double-buffering the accumulator in TMEM) to maximize overlap. This approach has been shown to be critical for achieving peak performance on the Blackwell architecture, often outperforming even highly optimized libraries like cuBLAS for certain matrix sizes.

#### Key Takeaways

*   Warp specialization assigns different tasks to different groups of warps within a kernel, enabling a more efficient, hardware-scheduled pipeline.
*   It is implemented in Gluon using `gl.warp_specialize`, with separate functions for each partition.
*   Synchronization between partitions relies heavily on `mbarrier`s.
*   This technique can effectively fuse the work of multiple programs into a single, highly coordinated kernel, maximizing hardware utilization and performance, especially on the latest GPU architectures.