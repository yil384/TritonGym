import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_cumsum_vector import triton_chunk_cumsum_vector
from benchmark.data_v2.torch.chunk_cumsum_vector import torch_chunk_cumsum_vector

def test_chunk_global_cumsum_vector():
    print("Testing chunk_global_cumsum_vector operator...")

    # Shapes for (B, H, T, S)
    input_shapes = [
        ((2, 3, 4, 5),),
        ((2, 3, 8, 10),),
        ((1, 2, 16, 32),),
        ((3, 2, 1, 8),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        s = torch.rand(shape[0], dtype=dtype, device=device)

        try:
            triton_out = triton_chunk_cumsum_vector(s)
            torch_out = torch_chunk_cumsum_vector(s)
            max_abs = (triton_out - torch_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"chunk_global_cumsum_vector case {idx}: B={shape[0][0]}, H={shape[0][1]}, T={shape[0][2]}, S={shape[0][3]} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"chunk_global_cumsum_vector case {idx}: B={shape[0][0]}, H={shape[0][1]}, T={shape[0][2]}, S={shape[0][3]}: {e}")


if __name__ == "__main__":
    test_chunk_global_cumsum_vector()
