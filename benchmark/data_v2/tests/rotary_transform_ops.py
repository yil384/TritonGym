# benchmark/data/tests/rotary_transform.py
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rotary_transform_ops import triton_rotary_transform_ops
from benchmark.data_v2.torch.rotary_transform_ops import torch_rotary_transform_ops

def test_rotary_transform():
    print("Testing rotary transform forward...")

    input_shapes = [
        ((2, 4, 3, 8), (4, 4), (4, 4)),  # B, S, H, D
        ((4, 3, 8, 8), (3, 4), (3, 4)),
        ((8, 2, 4, 16), (2, 8), (2, 8))
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        cos = torch.randn(shape[1], device=device, dtype=dtype)
        sin = torch.randn(shape[2], device=device, dtype=dtype)
        seqlen_offsets = 0

        try:
            torch_out = torch_rotary_transform_ops(x.clone(), cos, sin, seqlen_offsets)
            triton_out = triton_rotary_transform_ops(x.clone(), cos, sin, seqlen_offsets)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: {e}")

if __name__ == "__main__":
    test_rotary_transform()
