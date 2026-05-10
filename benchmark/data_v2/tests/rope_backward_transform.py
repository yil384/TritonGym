import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rope_backward_transform import triton_rope_backward_transform
from benchmark.data_v2.torch.rope_backward_transform import torch_rope_backward_transform

def test_rope_backward_func():
    print("Testing ROPE backward transformation...")

    input_shapes = [
        ((2, 8, 4, 16), (2, 8, 4, 16), (4, 8), (4, 8)),
        ((1, 4, 2, 8), (1, 4, 2, 8), (2, 4), (2, 4)),
        ((4, 16, 8, 32), (4, 16, 8, 32), (8, 16), (8, 16)),
        ((8, 32, 16, 64), (8, 32, 16, 64), (16, 32), (16, 32)),
        ((16, 64, 32, 128), (16, 64, 32, 128), (32, 64), (32, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        dq = torch.randn(shape[0], device=device, dtype=dtype)
        dk = torch.randn(shape[1], device=device, dtype=dtype)
        cos = torch.randn(shape[2], device=device, dtype=dtype)
        sin = torch.randn(shape[3], device=device, dtype=dtype)
        try:
            torch_out_dq = torch_rope_backward_transform(dq, dk, cos, sin)
            triton_out_dq = triton_rope_backward_transform(dq, dk, cos, sin)

            max_abs_dq = (torch_out_dq - triton_out_dq).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: dq max_abs={max_abs_dq:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: {e}")

if __name__ == "__main__":
    test_rope_backward_func()
