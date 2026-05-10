import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.triton.attn_gelu import triton_attn_gelu
from benchmark.data_ood_v2.torch.attn_gelu import torch_attn_gelu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def test_op():
    input_shapes =[
        ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64),),
        ((1, 4, 256, 64), (1, 4, 256, 64), (1, 4, 256, 64),),
        ((1, 8, 512, 64), (1, 8, 512, 64), (1, 8, 512, 64),),
        ((1, 16, 1024, 64), (1, 16, 1024, 64), (1, 16, 1024, 64),),
    ]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    for idx, shape in enumerate(input_shapes):
        
        q =torch.randn(shape[0], dtype=dtype, device=DEVICE)
        k = torch.randn(shape[1], dtype=dtype, device=DEVICE)
        v = torch.randn(shape[2], dtype=dtype, device=DEVICE)

        try:
            ref_out = torch_attn_gelu(q, k, v)
            tri_out = triton_attn_gelu(q, k, v)

            max_abs = torch.max(torch.abs(tri_out - ref_out)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except AssertionError:
            print(f"[FAIL] Test {idx} failed.")


if __name__ == "__main__":
    test_op()
