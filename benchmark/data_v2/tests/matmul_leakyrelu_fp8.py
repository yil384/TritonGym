import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.matmul_leakyrelu_fp8 import triton_matmul_leakyrelu_fp8
from benchmark.data_v2.torch.matmul_leakyrelu_fp8 import torch_matmul_leakyrelu_fp8

def test_matmul():
    print("Testing matmul_leakyrelu_fp8 operator...")
    input_shapes = [
        ((256, 64), (64, 256)),
        ((512, 128), (128, 512)),
        ((1024, 64), (64, 1024)),
        ((128, 128), (128, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, (Ashape, Bshape) in enumerate(input_shapes):
        a = torch.randn(*Ashape, device=device, dtype=dtype)
        b = torch.randn(*Bshape, device=device, dtype=dtype)
        try:
            # Triton output
            triton_out = triton_matmul_leakyrelu_fp8(a, b)
            # Torch output
            torch_out = torch_matmul_leakyrelu_fp8(a, b)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"matmul case {idx}: A{Ashape} x B{Bshape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"matmul case {idx}: A{Ashape} x B{Bshape}: {e}")

if __name__ == "__main__":
    test_matmul()
