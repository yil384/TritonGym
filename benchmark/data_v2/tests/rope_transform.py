import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rope_transform import triton_rope_transform
from benchmark.data_v2.torch.rope_transform import torch_rope_transform

def test_rope_transform():
    print("Testing RoPE forward...")

    input_shapes = [
        ((2, 4, 8, 16), (2, 4, 8, 16), (8, 8), (8, 8)),
        ((1, 2, 4, 8), (1, 2, 4, 8), (4, 4), (4, 4)),
        ((4, 8, 16, 32), (4, 8, 16, 32), (16, 16), (16, 16)),
        ((8, 16, 32, 64), (8, 16, 32, 64), (32, 32), (32, 32)),
        ((16, 32, 64, 128), (16, 32, 64, 128), (64, 64), (64, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        cos = torch.randn(shape[2], device=device, dtype=dtype)
        sin = torch.randn(shape[3], device=device, dtype=dtype)

        try:
            torch_q = torch_rope_transform(q, k, cos, sin)
            triton_q = triton_rope_transform(q, k, cos, sin)

            max_abs_q = (torch_q - triton_q).abs().max().item()

            print(f"\033[92m OK \033[0m case {idx}: Q max_abs={max_abs_q:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: {e}")


if __name__ == "__main__":
    test_rope_transform()
