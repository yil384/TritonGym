import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.gluon_add import triton_gluon_add
from benchmark.data_dsl_v2.torch.gluon_add import torch_gluon_add


def test_elementwise_add():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shapes =[
        ((1000, 2000), (1000, 2000),), 
        ((4000, 120), (4000, 120),),
        ((8192, 512), (8192, 512),),
        ((16384, 256), (16384, 256),),
        ((32768, 128), (32768, 128),),
        ((65536, 64), (65536, 64),),
        ((131072, 32), (131072, 32),),
        ((262144, 16), (262144, 16),),
        ((524288, 8), (524288, 8),),
        ((1048576, 4), (1048576, 4),),
    ]
    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device)
        b = torch.randn(shape[1], device=device)

        c_triton = triton_gluon_add(a, b)
        c_torch = torch_gluon_add(a, b)
        try:
            max_abs = torch.max(torch.abs(c_triton - c_torch)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except AssertionError:
            print(f"[FAIL] Test {idx} failed.") 

if __name__ == "__main__":
    test_elementwise_add()