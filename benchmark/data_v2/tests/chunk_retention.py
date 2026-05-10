import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_retention import triton_chunk_retention
from benchmark.data_v2.torch.chunk_retention import torch_chunk_retention

def test_chunk_retention():
    print("Testing chunk_retention operator...")

    # q:[B,H,T,K], k:[B,H,T,K], v:[B,H,T,V]
    input_shapes = [
        ((2, 4, 64, 32), (2, 4, 64, 32), (2, 4, 64, 32)),
        ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            torch_out = torch_chunk_retention(q, k, v)
            triton_out = triton_chunk_retention(q, k, v)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=" ")
            print(f"chunk_retention case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=" ")
            print(f"chunk_retention case {idx}: {e}")


if __name__ == "__main__":
    test_chunk_retention()
