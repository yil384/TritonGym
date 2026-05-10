# benchmark/data/tests/rotary_transform_test.py
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rotary_transform import triton_rotary_transform
from benchmark.data_v2.torch.rotary_transform import torch_rotary_transform

def test_rotary_transform():
    print("Testing rotary_transform...")

    input_shapes = [
        ((2, 128, 4, 64), (128, 16), (128, 16)),  # with seqlen_offsets
        ((3, 64, 8, 32), (64, 16), (64, 16)),
        ((1, 256, 2, 128), (256, 32), (256, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        cos = torch.randn(shape[1], device=device, dtype=dtype)
        sin = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            torch_out = torch_rotary_transform(x.clone(), cos, sin)
            triton_out = triton_rotary_transform(x.clone(), cos, sin)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: {e}")

if __name__ == "__main__":
    test_rotary_transform()
