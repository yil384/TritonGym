import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

@gluon.jit
def issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # Track completion of both TMA reads with the same mbarrier.
    yoff = copy_index * YBLOCK
    bar = bars.index(copy_index % num_buffers)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [xoff, yoff], bar, a_smem.index(copy_index % num_buffers))
    tma.async_copy_global_to_shared(b_desc, [xoff, yoff], bar, b_smem.index(copy_index % num_buffers))
    return copy_index + 1

@gluon.jit
def perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout: gl.constexpr, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # Wait for the copy from num_buffers-1 iterations ago to complete.
    read_phase = read_index // num_buffers & 1
    mbarrier.wait(bars.index(read_index % num_buffers), read_phase)
    a_val = a_smem.index(read_index % num_buffers).load(layout)
    b_val = b_smem.index(read_index % num_buffers).load(layout)
    c_val = a_val + b_val
    yoff = read_index * YBLOCK
    # Pipeline the store by rotating the store wait.
    tma.store_wait(pendings=0)
    c_smem.store(c_val)
    fence_async_shared()
    # Issue the store without waiting for it.
    tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_smem)
    return read_index + 1

@gluon.jit
def elementwise_add_tma_kernel(  #
        a_desc, b_desc, c_desc, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoff = pid * XBLOCK

    dtype: gl.constexpr = a_desc.type.block_type.element_ty
    # Allocate multibuffered shared memory for the input buffers.
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], b_desc.layout)

    # Allocate shared memory for the TMA store.
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], c_desc.layout)

    # Allocate mbarriers to track completion of the TMA reads.
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    copy_index = 0
    read_index = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)

    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for _ in gl.static_range(num_buffers - 1):
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(bars.index(i))

    # Wait for the last store to complete.
    tma.store_wait(pendings=0)


def triton_gluon_add(a, b, XBLOCK=32, YBLOCK=64, num_buffers=2):
    c = torch.empty_like(a, device=a.device)
    assert a.shape == b.shape == c.shape
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    block_shape = [XBLOCK, YBLOCK]
    # TMA descriptors require NVMMASharedLayout.
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # The strides of TMA descriptors must be 16-byte aligned.
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)
    elementwise_add_tma_kernel[grid](a_desc, b_desc, c_desc, xnumel, ynumel, XBLOCK, YBLOCK, num_buffers)
    return c