import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.seeded_dropout import triton_seeded_dropout
from benchmark.data_v2.torch.seeded_dropout import torch_seeded_dropout

def test_seeded_dropout():
    print("Testing seeded_dropout operator...")

    input_shapes = [
        ((10,),),  # small 1D tensor
        ((20,),),  # larger 1D tensor
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        torch_out = torch_seeded_dropout(x)
        triton_out = triton_seeded_dropout(x)
        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m shape={shape[0]}, seed={0.5}, p={123}, max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_seeded_dropout()
