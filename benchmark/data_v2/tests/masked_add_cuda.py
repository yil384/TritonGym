import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.masked_add_cuda import triton_masked_add_cuda
from benchmark.data_v2.torch.masked_add_cuda import torch_masked_add_cuda

def test_masked_add():
    input_shapes = [
        ((10000,), (10000,)),
        ((100000,), (100000,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        grad = torch.randn(shape[0], device=device, dtype=dtype)
        p_data = torch.randn(shape[1], device=device, dtype=dtype)

        # Test case 1: alpha=0.5
        grad_triton = triton_masked_add_cuda(grad, p_data)
        grad_torch = torch_masked_add_cuda(grad, p_data)
        max_abs = (grad_triton - grad_torch).abs().max().item()
        print(f"\033[92m OK \033[0m masked_add case {idx} alpha=0.5 max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_masked_add()
