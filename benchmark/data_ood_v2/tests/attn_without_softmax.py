import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.triton.attn_without_softmax import triton_attn_without_softmax
from benchmark.data_ood_v2.torch.attn_without_softmax import torch_attn_without_softmax

def test_op():
    input_shapes = [
        ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),
        ((4, 2, 128, 64), (4, 2, 128, 64), (4, 2, 128, 64)),
    ]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], dtype=dtype, device=DEVICE)
        k = torch.randn(shape[1], dtype=dtype, device=DEVICE)
        v = torch.randn(shape[2], dtype=dtype, device=DEVICE)

        try:
            triton_out = triton_attn_without_softmax(q, k, v)
            torch_out = torch_attn_without_softmax(q, k, v)
            max_diff = (triton_out - torch_out).abs().max().item()
            print(f"[OK] Test case {idx}: max absolute difference = {max_diff}")
        except Exception as e:
            print(f"[FAIL] Test case {idx} raised an exception: {e}")


if __name__ == "__main__":
    test_op()