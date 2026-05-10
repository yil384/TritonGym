import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.batched_vecmat_mult import triton_batched_vecmat_mult
from benchmark.data_v2.torch.batched_vecmat_mult import torch_batched_vecmat_mult

def test_batched_vecmat():
    print("Testing batched vecmat operator...")

    # Shapes: (M, N, K, block_m, block_n, block_k)
    input_shapes = [
        ((128, 128), (128, 128, 128)),
        ((64, 64), (64, 64, 64)),
    ]

    for idx, shape in enumerate(input_shapes):
        A = torch.randn(shape[0], device='cuda', dtype=torch.float32)  # shape: [M, K]
        B = torch.randn(shape[1], device='cuda', dtype=torch.float32)  # shape: [M, N, K]

        try:
            # Torch reference
            torch_out = torch_batched_vecmat_mult(A, B)

            # Triton kernel
            triton_out = triton_batched_vecmat_mult(A, B)

            # Difference
            max_abs = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=" ")
            print(f"vecmat case {idx}: M={shape[1][0]}, N={shape[1][1]}, K={shape[1][2]}, max_abs={max_abs:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=" ")
            print(f"vecmat case {idx}: M={shape[1][0]}, N={shape[1][1]}, K={shape[1][2]}: {e}")


if __name__ == "__main__":
    test_batched_vecmat()
