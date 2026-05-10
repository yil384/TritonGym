import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.sin_computation import triton_sin_computation
from benchmark.data_v2.torch.sin_computation import torch_sin_computation

def test_sin():
    print("Testing sin operator...")

    input_shapes = [
        ((4,),),
        ((8,),),
        ((16,),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], dtype=dtype, device=device)

        # Compute outputs
        out_torch = torch_sin_computation(x)
        out_triton = triton_sin_computation(x)

        max_abs = (out_torch - out_triton).abs().max().item()
        print(f"\033[92m OK \033[0m sin case {idx}: shape={shape[0]}, max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_sin()
