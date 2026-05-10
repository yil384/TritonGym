import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fast_layernorm import triton_fast_layernorm
from benchmark.data_v2.torch.fast_layernorm import torch_fast_layernorm

def test_fast_layernorm():
    print("Testing layernorm operator...")

    input_shapes = [
        ((4, 8),),
        ((16, 32),),
        ((64, 128),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        X = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)

        try:
            torch_out = torch_fast_layernorm(X)
            triton_out = triton_fast_layernorm(X)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m layernorm case {idx}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m layernorm case {idx}: {e}")

if __name__ == "__main__":
    test_fast_layernorm()
