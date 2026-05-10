import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.masked_select import triton_masked_select
from benchmark.data_v2.torch.masked_select import torch_masked_select

def test_masked_select():
    input_shapes = [
        ((4, 4),),
        ((2, 3, 4),),
        ((2, 2, 2, 2),),
        ((512, 1024),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32


    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        torch_out = torch_masked_select(x)
        triton_out = triton_masked_select(x)
        
        # Validate max absolute difference
        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m masked_select case {idx} max_abs={max_abs:.3e}")


if __name__ == "__main__":
    test_masked_select()
