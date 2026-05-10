import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_gla_simple import triton_chunk_gla_simple
from benchmark.data_v2.torch.chunk_gla_simple import torch_chunk_gla_simple

def test_chunk_gla_simple():
    print("Testing chunk_gla_simple operator...")

    # q:[B,H,T,K], k:[B,H,T,K], v:[B,H,T,V], h:[B,H,NT*K,V], g:[B,H,T]
    # With BT=32 (default), NT = T//BT
    # Shape order in tuples: (q, k, v, h, g)
    input_shapes = [
        ((2, 4, 64, 32), (2, 4, 64, 32), (2, 4, 64, 32), (2, 4, 64, 32), (2, 4, 64)),   # NT=2, NT*K=64
        ((1, 2, 32, 16), (1, 2, 32, 16), (1, 2, 32, 16), (1, 2, 16, 16), (1, 2, 32)),   # NT=1, NT*K=16
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        # Create inputs
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)
        h = torch.randn(shape[3], device=device, dtype=dtype)
        g = torch.randn(shape[4], device=device, dtype=dtype)

        try:
            # Triton output
            triton_out = triton_chunk_gla_simple(h, q, k, v, g)

            # Torch output: simplified verification, per batch/head
            torch_out = torch_chunk_gla_simple(h, q, k, v, g)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"chunk_gla_simple case {idx}: shape {shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"chunk_gla_simple case {idx}: shape {shape}: {e}")

if __name__ == "__main__":
    test_chunk_gla_simple()
