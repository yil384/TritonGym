import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.relu import triton_relu
from benchmark.data_v2.torch.relu import torch_relu

def test_relu():
    print("Testing relu operator...")

    input_shapes = [
        ((5,),),          # small vector
        ((10,),),         # slightly larger
        ((1024,),),       # block-sized
        ((2048,),),       # large vector
        ((10000,),),      # very large
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_relu(a)
            triton_out = triton_relu(a)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"relu case {idx}: input{shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"relu case {idx}: input{shape}: {e}")

if __name__ == "__main__":
    test_relu()
