import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_cumsum_kernel import triton_chunk_cumsum_kernel
from benchmark.data_v2.torch.chunk_cumsum_kernel import torch_chunk_cumsum_kernel 

def test_chunk_global_cumsum_scalar():
    print("Testing chunk_global_cumsum_scalar operator...")

    # Shapes for (B, H, T)
    input_shapes = [
        ((2, 3, 4),),
        ((4, 2, 8),),
        ((1, 5, 10),),
        ((3, 3, 16),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        s = torch.randn(shape[0], dtype=dtype, device=device)

        try:
            triton_out = triton_chunk_cumsum_kernel(s)
            torch_out = torch_chunk_cumsum_kernel(s)
            max_abs = (triton_out - torch_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"chunk_global_cumsum_scalar case {idx}: B={shape[0][0]}, H={shape[0][1]}, T={shape[0][2]} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"chunk_global_cumsum_scalar case {idx}: B={shape[0][0]}, H={shape[0][1]}, T={shape[0][2]}: {e}")


if __name__ == "__main__":
    test_chunk_global_cumsum_scalar()
