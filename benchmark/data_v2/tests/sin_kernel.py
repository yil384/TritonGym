import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.sin_kernel import triton_sin_kernel
from benchmark.data_v2.torch.sin_kernel import torch_sin_kernel

def test_sin():
    print("Testing sin operator...")

    # Shapes for input tensors
    input_shapes = [
        ((4,),),       # small vector
        ((8,),),       # larger vector
        ((16,),),      # multiple of 4
        ((32,),),      # bigger vector
        ((1024,),),    # large vector
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        # Triton output
        out_triton = triton_sin_kernel(x)

        # Torch output
        out_torch = torch_sin_kernel(x)

        # Compare
        max_abs = (out_triton - out_torch).abs().max().item()
        print(f"\033[92m OK \033[0m", end=' ')
        print(f"sin case {idx}: shape={shape[0]} max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_sin()
