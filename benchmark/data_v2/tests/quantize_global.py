import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.quantize_global import triton_quantize_global
from benchmark.data_v2.torch.quantize_global import torch_quantize_global

def test_quantize_global():
    print("Testing quantize_global operator...")
    
    input_shapes = [
        ((2048,),),
        ((1024,),),
        ((3072,),),
        ((4096,),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_quantize_global(x)
            triton_out = triton_quantize_global(x)
            
            max_abs_diff = (torch_out.float() - triton_out.float()).abs().max().item()
            print(f"\033[92m OK \033[0m quantize_global case {idx}: input{shape} max_abs_diff={max_abs_diff:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m quantize_global case {idx}: input{shape}: {e}")

if __name__ == "__main__":
    test_quantize_global()
