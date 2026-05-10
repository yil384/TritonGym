import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.reversed_cumsum import triton_reversed_cumsum
from benchmark.data_v2.torch.reversed_cumsum import torch_reversed_cumsum

def test_reversed_cumsum_vector():
    print("Testing reversed_cumsum_vector operator...")

    input_shapes = [
        ((2, 3, 4, 5),),
        ((1, 1, 8, 8),),
        ((4, 2, 16, 16),),
        ((3, 3, 32, 32),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_reversed_cumsum(a)
            triton_out = triton_reversed_cumsum(a)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"reversed_cumsum_vector case {idx}: input{shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"reversed_cumsum_vector case {idx}: input{shape}: {e}")

if __name__ == "__main__":
    test_reversed_cumsum_vector()
