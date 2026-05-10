import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.f8_to_f16 import triton_f16_to_f8, triton_f8_to_f16
from benchmark.data_v2.torch.f8_to_f16 import torch_f16_to_f8, torch_f8_to_f16

def test_f8_f16():
    print("Testing f16<->f8 conversion operator...")

    input_shapes = [
        ((16, 128),),
        ((32, 64),),
        ((64, 32),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], dtype=dtype, device=device) # scale to better test int8 range
        o_torch_f8 = torch_f16_to_f8(a)
        o_triton_f8 = triton_f16_to_f8(a)
        
        max_abs = (o_torch_f8 - o_triton_f8).abs().max().item()
        print(f"\033[92m OK \033[0m f16<->f8 case {idx}, max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_f8_f16()
