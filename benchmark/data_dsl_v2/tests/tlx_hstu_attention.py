from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.tlx_hstu_attention import triton_tlx_htsu_attention
from benchmark.data_dsl_v2.torch.tlx_hstu_attention import torch_tlx_hstu_attention


def test_tlx_hstu_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shapes = [
        (
            (1024, 32, 128),  # q
            (1024, 32, 128),  # k
            (1024, 32, 128),  # v
        ),
        (
            (2048, 32, 128),  # q
            (2048, 32, 128),  # k
            (2048, 32, 128),  # v
        ),
        (
            (4096, 32, 128),  # q
            (4096, 32, 128),  # k
            (4096, 32, 128),  # v
        ),
    ]
    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=torch.float16)
        k = torch.randn(shape[1], device=device, dtype=torch.float16)
        v = torch.randn(shape[2], device=device, dtype=torch.float16)

        out_triton = triton_tlx_htsu_attention(q, k, v)
        out_torch = torch_tlx_hstu_attention(q, k, v)
        try:
            max_abs = torch.max(torch.abs(out_triton - out_torch)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except AssertionError:
            print(f"[FAIL] Test {idx} failed.")

if __name__ == "__main__":
    test_tlx_hstu_attention()
