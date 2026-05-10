# benchmark/data/tests/generative_affine_matmul.py
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.triton.generative_affine_matmul import triton_generative_affine_matmul
from benchmark.data_ood_v2.torch.generative_affine_matmul import torch_generative_affine_matmul

def test_generative_affine_matmul():
    print("Testing generative_affine_matmul operator...")

    input_shapes = [
        ((128, 128),),
        ((256, 512),),
        ((512, 1024),),
        ((1024, 2048),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_generative_affine_matmul(a)
            triton_out = triton_generative_affine_matmul(a)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m generative_affine_matmul case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m generative_affine_matmul case {idx}: {e}")

if __name__ == "__main__":
    test_generative_affine_matmul()
