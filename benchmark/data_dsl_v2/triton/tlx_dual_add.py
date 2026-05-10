import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

@triton.jit
def tlx_dual_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    with tlx.async_tasks():
        with tlx.async_task("default"):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(z_ptr + offsets, output, mask=mask)
        with tlx.async_task(num_warps=4):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            output = a + b
            tl.store(c_ptr + offsets, output, mask=mask)


def triton_tlx_dual_add(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    output1 = torch.empty_like(x)
    output2 = torch.empty_like(a)
    assert x.device == DEVICE and y.device == DEVICE and output1.device == DEVICE
    n_elements = output1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    tlx_dual_add_kernel[grid](x, y, output1, a, b, output2, n_elements, BLOCK_SIZE=1024)
    return output1, output2