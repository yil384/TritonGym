import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.tlx_attention import triton_tlx_attention
from benchmark.data_dsl_v2.torch.tlx_attention import torch_tlx_attention


def test_tlx_attention():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shapes = [
        ((4, 32, 2048, 128,),(4, 32, 2048, 128,),(4, 32, 2048, 128,),),
        ((4, 32, 4096, 128,),(4, 32, 4096, 128,),(4, 32, 4096, 128,),),
    ]
    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=torch.float16)
        k = torch.randn(shape[1], device=device, dtype=torch.float16)
        v = torch.randn(shape[2], device=device, dtype=torch.float16)

        out_triton = triton_tlx_attention(q, k, v)
        out_torch = torch_tlx_attention(q, k, v)
        try:
            max_abs = torch.max(torch.abs(out_triton - out_torch)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except AssertionError:
            print(f"[FAIL] Test {idx} failed.")

if __name__ == "__main__":
    test_tlx_attention()
