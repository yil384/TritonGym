import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.multinomial_sampling import triton_multinomial_sampling
from benchmark.data_v2.torch.multinomial_sampling import torch_multinomial_sampling

def test_multinomial_sampling():
    print("Testing multinomial_sampling operator...")

    input_shapes = [
        ((2, 3),),  # batch=2, tokens=3
        ((4, 5),),
        ((2, 1),),
        ((3, 10),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        scores = torch.rand(shape[0], device=device, dtype=dtype)
  
        triton_out = triton_multinomial_sampling(scores)
        torch_out = torch_multinomial_sampling(scores)
        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m", end=' ')
        print(f"multinomial_sampling case {idx}: shape={shape}, max_abs={max_abs:.3e}")


if __name__ == "__main__":
    test_multinomial_sampling()
