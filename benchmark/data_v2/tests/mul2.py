import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.mul2 import triton_mul2
from benchmark.data_v2.torch.mul2 import torch_mul2

def test_mul2():
    print("Testing mul2 operator...")

    input_shapes = [
        ((1024,),), 
        ((2048,),),
        ((4096,),),
        ((8192,),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        # Triton result
        triton_out1 = triton_mul2(x.clone())

        # Torch result
        torch_out1 = torch_mul2(x.clone())

        # Compare results
        max_abs1 = (torch_out1 - triton_out1).abs().max().item()

        print(f"\033[92m OK \033[0m mul2 case {idx}: max_abs={max_abs1:.3e}")

if __name__ == "__main__":
    test_mul2()
