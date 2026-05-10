import math

import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.tlx_simplicial_attention_pipelined import triton_tlx_simplicial_attention_pipelined
from benchmark.data_dsl_v2.torch.tlx_simplicial_attention_pipelined import torch_tlx_simplicial_attention_pipelined


def test_tlx_simplicial_attention_pipelined():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shapes = [
        ((2, 32, 128, 128),(2, 32, 128, 128),(2, 32, 128, 128),(2, 32, 128, 128),(2, 32, 128, 128),),
        ((4, 32, 128, 128),(4, 32, 128, 128),(4, 32, 128, 128),(4, 32, 128, 128),(4, 32, 128, 128),),
    ]
    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=torch.bfloat16)
        k1 = torch.randn(shape[1], device=device, dtype=torch.bfloat16)
        k2 = torch.randn(shape[2], device=device, dtype=torch.bfloat16)
        v1 = torch.randn(shape[3], device=device, dtype=torch.bfloat16)
        v2 = torch.randn(shape[4], device=device, dtype=torch.bfloat16)

        out_triton = triton_tlx_simplicial_attention_pipelined(q, k1, k2, v1, v2)
        out_torch = torch_tlx_simplicial_attention_pipelined(q, k1, k2, v1, v2)
        try:
            max_abs = torch.max(torch.abs(out_triton[0] - out_torch)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except AssertionError:
            print(f"[FAIL] Test {idx} failed.")

if __name__ == "__main__":
    test_tlx_simplicial_attention_pipelined()
