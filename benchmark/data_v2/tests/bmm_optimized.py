import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.bmm_optimized import triton_bmm_optimized
from benchmark.data_v2.torch.bmm_optimized import torch_bmm_optimized

def test_bmm():
    print("Testing bmm operator...")

    input_shapes = [
        ((1, 32, 32), (1, 32, 32)),
        ((8, 64, 32), (8, 32, 64)),
        ((16, 128, 32), (16, 32, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # BMM typically uses float32 for numerical testing

    for idx, shape in enumerate(input_shapes):
        A = torch.randn(shape[0], device=device, dtype=dtype)
        B = torch.randn(shape[1], device=device, dtype=dtype)

        try:
            torch_out = torch_bmm_optimized(A, B)
            triton_out = triton_bmm_optimized(A, B)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"bmm case {idx}: batch={shape[0][0]}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"bmm case {idx}: batch={shape[0][0]}: {e}")


if __name__ == "__main__":
    test_bmm()
