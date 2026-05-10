
import triton
import triton.language as tl
import torch

@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask).to(tl.float32)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)

def triton_sin_computation(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sin_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out



##################################################################################################################################################


def test_sin_triton():
    results = {}
    
    # Test case 1
    x1 = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
    out1 = torch.empty_like(x1)
    triton_sin_computation(x1)
    results['test_case_1'] = out1

    # Test case 2
    x2 = torch.tensor([4.0, 5.0, 6.0, 7.0], device='cuda')
    out2 = torch.empty_like(x2)
    triton_sin_computation(x2)
    results['test_case_2'] = out2

    # Test case 3
    x3 = torch.tensor([8.0, 9.0, 10.0, 11.0], device='cuda')
    out3 = torch.empty_like(x3)
    triton_sin_computation(x3)
    results['test_case_3'] = out3

    # Test case 4
    x4 = torch.tensor([12.0, 13.0, 14.0, 15.0], device='cuda')
    out4 = torch.empty_like(x4)
    triton_sin_computation(x4)
    results['test_case_4'] = out4

    return results

result_gold = test_sin_triton()
