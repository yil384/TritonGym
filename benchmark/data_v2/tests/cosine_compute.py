# benchmark/data/tests/cos_func_test.py
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.cosine_compute import triton_cosine_compute
from benchmark.data_v2.torch.cosine_compute import torch_cosine_compute

def test_cos_func():
    print("Testing cosine function...")

    input_shapes = [
        ((1024,),),
        ((2048,),),
        ((4096,),),
        ((8192,),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.rand(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_cosine_compute(x)
            triton_out = triton_cosine_compute(x)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}, {e}")

if __name__ == "__main__":
    test_cos_func()
