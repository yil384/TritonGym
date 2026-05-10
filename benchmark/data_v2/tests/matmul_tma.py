import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.matmul_tma import triton_matmul_tma
from benchmark.data_v2.torch.matmul_tma import torch_matmul_tma

def test_matmul_tma():
    print("Testing matmul TMA operator...")

    input_shapes = [
        ((128, 128), (128, 128)),
        ((128, 256), (256, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=torch.float16)
        b = torch.randn(shape[1], device=device, dtype=torch.float16)

        try:
            torch_out = torch_matmul_tma(a, b)
            triton_out = triton_matmul_tma(a, b)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m matmul case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m matmul case {idx}: {e}")

if __name__ == "__main__":
    test_matmul_tma()
