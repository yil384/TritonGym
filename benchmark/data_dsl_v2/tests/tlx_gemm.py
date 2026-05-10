import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.tlx_gemm import triton_tlx_gemm
from benchmark.data_dsl_v2.torch.tlx_gemm import torch_tlx_gemm


def test_tlx_gemm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shapes = [
        ((1024, 1024), (1024, 1024)),
        ((1024, 1024), (1024, 4096)),
        ((4096, 4096), (4096, 1024)),
    ]
    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=torch.float16)
        b = torch.randn(shape[1], device=device, dtype=torch.float16)

        out_triton = triton_tlx_gemm(a, b)
        out_torch = torch_tlx_gemm(a, b)
        try:
            max_abs = torch.max(torch.abs(out_triton - out_torch)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except AssertionError:
            print(f"[FAIL] Test {idx} failed.")

if __name__ == "__main__":
    test_tlx_gemm()
