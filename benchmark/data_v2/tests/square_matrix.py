import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.square_matrix import triton_square_matrix
from benchmark.data_v2.torch.square_matrix import torch_square_matrix

def test_square_matrix():
    print("Testing square operator...")
    input_shapes = [
        ((128, 64),),
        ((128, 128),),
        ((128, 256),),
        ((128, 512),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_square_matrix(a)
            triton_out = triton_square_matrix(a)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"square case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"square case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_square_matrix()
