import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.iv_dependent_matmul import triton_iv_dependent_matmul
from benchmark.data_v2.torch.iv_dependent_matmul import torch_iv_dependent_matmul

def test_iv_dependent_matmul():
    print("Testing IV Dependent MatMul operator...")

    # Shapes for (M, K) x (K, N)
    input_shapes = [
        ((256, 256), (256, 256)),
        ((512, 128), (128, 512)),
        ((1024, 512), (512, 1024)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32

    for idx, (Ashape, Bshape) in enumerate(input_shapes):
        a = torch.randn(*Ashape, device=device, dtype=dtype)
        b = torch.randn(*Bshape, device=device, dtype=dtype)
        try:
            torch_out = torch_iv_dependent_matmul(a, b)
            triton_out = triton_iv_dependent_matmul(a, b)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"matmul case {idx}: A{Ashape} x B{Bshape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"matmul case {idx}: A{Ashape} x B{Bshape}: {e}")

if __name__ == "__main__":
    test_iv_dependent_matmul()
