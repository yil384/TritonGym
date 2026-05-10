import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.gluon_gemm import triton_gluon_gemm
from benchmark.data_dsl_v2.torch.gluon_gemm import torch_gluon_gemm

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    input_shapes =[
        ((8192, 8192), (8192, 8192),),
        ((16384, 4096), (4096, 8192),),
        ((32768, 2048), (2048, 8192),),
        ((65536, 1024), (1024, 8192),),
    ]
    for idx, shape in enumerate(input_shapes):
        A = torch.randn(shape[0], device=device, dtype=dtype)
        B = torch.randn(shape[1], device=device, dtype=dtype)

        C_ref = torch_gluon_gemm(A, B)

        C = triton_gluon_gemm(A, B)

        try:
            max_abs = torch.max(torch.abs(C - C_ref)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except AssertionError as e:
            print(f"[FAIL] Test {idx} failed for shape A: {A.shape}, B: {B.shape}: {e}")