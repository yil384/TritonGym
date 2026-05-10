import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.pow_scalar_tensor import triton_pow_scalar_tensor
from benchmark.data_v2.torch.pow_scalar_tensor import torch_pow_scalar_tensor

def test_pow_scalar_tensor():
    print("Testing pow_scalar_tensor operator...")

    input_shapes = [
        ((4,),),
        ((7,),),
        ((1024,),),
        ((4096,),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)

        try:
            torch_out = torch_pow_scalar_tensor(a)
            triton_out = triton_pow_scalar_tensor(a)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_pow_scalar_tensor()
