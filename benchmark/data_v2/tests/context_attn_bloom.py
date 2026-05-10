# benchmark/data/tests/context_attn_bloom.py

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.context_attn_bloom import triton_context_attn_bloom
from benchmark.data_v2.torch.context_attn_bloom import torch_context_attn_bloom


def test_context_attn_bloom():
    print("Testing context_attn_bloom operator...")

    # input_shapes: [(B, H, T, D), extra_kv_tokens]
    input_shapes = [
        ((1, 4, 128, 64), 32),
        ((2, 8, 256, 96), 64),
    ]

    input_shapes = [
        ((128, 4, 64), (128 + 32, 2, 64), (128 + 32, 2, 64)),
        ((256, 8, 96), (256 + 64, 4, 96), (256 + 64, 4, 96)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            torch_out = torch_context_attn_bloom(q, k, v)
            triton_out = triton_context_attn_bloom(q, k, v)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m context_attn_bloom case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m context_attn_bloom case {idx}: {e}")


if __name__ == "__main__":
    test_context_attn_bloom()
