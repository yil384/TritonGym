import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.matmul import triton_matmul
from benchmark.data_v2.torch.matmul import torch_matmul

def test_matmul():
    # Minimal benchmark harness for torch_matmul without context object.

    # Each entry indicates the shape for each tensor (A, B):
    # e.g., [((M, K), (K, N)), ...]
    # Back-compat: entries like (M, K, N) are converted to ((M, K), (K, N)).
    input_shapes = [
        ((1024, 1024), (1024, 1024)),
        ((1024, 1024), (1024, 4096)),
        ((4096, 4096), (4096, 1024)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    # Default: no transpose

    for entry in input_shapes:
        Ashape, Bshape =entry
        a = torch.randn(*Ashape, device=device, dtype=dtype)
        b = torch.randn(*Bshape, device=device, dtype=dtype)

        torch_out = torch_matmul(a, b)
        triton_out = triton_matmul(a, b)
        max_abs = (torch_out - triton_out).abs().max().item()

        print(f"torch_matmul A{Ashape} x B{Bshape}: max_abs={max_abs:.3e}")


if __name__ == "__main__":
    test_matmul()