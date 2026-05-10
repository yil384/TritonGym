import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.dequantize_matmul import triton_dequantize_matmul
from benchmark.data_v2.torch.dequantize_matmul import torch_dequantize_matmul

def test_dequantize_matmul():
    print("Testing dequantize_matmul...")

    input_shapes = [
        ((64, 128), (128, 256), (256,)),
        ((128, 64), (64, 128), (128,)),
        ((32, 32), (32, 32), (32,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        b = torch.randn(shape[1], device=device, dtype=dtype)
        b_scale = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            torch_out = torch_dequantize_matmul(a, b, b_scale)
            triton_out = triton_dequantize_matmul(a, b, b_scale)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: A{shape[0]} x B{shape[1]}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: A{shape[0]} x B{shape[1]}: {e}")

if __name__ == "__main__":
    test_dequantize_matmul()
