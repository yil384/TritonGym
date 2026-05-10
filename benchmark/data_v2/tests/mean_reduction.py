import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.mean_reduction import triton_mean_reduction
from benchmark.data_v2.torch.mean_reduction import torch_mean_reduction

def test_mean_reduction():
    print("Testing mean_reduction operator...")

    # Define test shapes and reduction dimensions
    input_shapes = [
        ((2, 3, 4, 5),), 
        ((4, 5, 6, 7, 8),), 
        ((8, 7, 6, 5, 4, 3),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=torch.float16)

        try:
            torch_out = torch_mean_reduction(x)
            triton_out = triton_mean_reduction(x)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m mean_reduction case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m mean_reduction case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_mean_reduction()
