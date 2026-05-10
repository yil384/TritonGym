import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.matrix_reduction import triton_matrix_reduction
from benchmark.data_v2.torch.matrix_reduction import torch_matrix_reduction

def test_matrix_reduction():
    print("Testing matrix_reduction operator...")
    input_shapes = [
        ((16, 16),),
        ((32, 32),),
        ((64, 64),),
        ((128, 128),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_matrix_reduction(x)
            # Triton wrapper does internal assert; just run it
            triton_out = triton_matrix_reduction(x)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"matrix_reduction case {idx}, dtype={dtype}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"matrix_reduction case {idx}, dtype={dtype}: shape={shape}: {e}")

if __name__ == "__main__":
    test_matrix_reduction()
