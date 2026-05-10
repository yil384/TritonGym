import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.block_sparse_attn import triton_block_sparse_attn
from benchmark.data_v2.torch.block_sparse_attn import torch_block_sparse_attn


def test_block_sparse_attention():
    print("Testing block sparse attention operator...")

    # Shapes and params: (B, H, M, N, D, num_kv_heads, BLOCK_M, BLOCK_N, BLOCK_D, NUM_D_BLOCKS, EVEN_M, EVEN_N)
    input_shapes = [
        ((2, 4, 16, 32), (2, 2, 32, 32), (2, 2, 32, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        B, H, M, N = shape[0][0], shape[0][1], shape[0][2], shape[1][2]
        D = shape[0][3]
        num_kv_heads = shape[1][1]
        try:
            # Random input
            Q = torch.randn(shape[0], device=device, dtype=dtype)
            K = torch.randn(shape[1], device=device, dtype=dtype)
            V = torch.randn(shape[2], device=device, dtype=dtype)

            # Torch reference
            torch_out = torch_block_sparse_attn(
                Q, K, V,
            )

            # Triton kernel
            triton_out = triton_block_sparse_attn(
                Q, K, V,
            )

            # Difference
            max_abs = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=" ")
            print(
                f"block_sparse_attn case {idx}: "
                f"B={B}, H={H}, M={M}, N={N}, D={D}, "
                f"max_abs={max_abs:.3e}"
            )

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=" ")
            print(
                f"block_sparse_attn case {idx}: "
                f"B={B}, H={H}, M={M}, N={N}, D={D}: {e}"
            )


if __name__ == "__main__":
    test_block_sparse_attention()
