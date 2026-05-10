# benchmark/data/tests/rotary_emb.py
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rotary_emb import triton_rotary_emb
from benchmark.data_v2.torch.rotary_emb import torch_rotary_emb

def test_rotary_emb():
    print("Testing rotary embedding forward...")

    input_shapes = [
        ((32, 8, 64), (32, 4, 64), (32, 64), (32, 64)),  # total_len, head_num_q, head_dim
        ((16, 4, 32), (16, 2, 32), (16, 32), (16, 32)),
        ((64, 16, 128), (64, 8, 128), (64, 128), (64, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        cos = torch.randn(shape[2], device=device, dtype=dtype)
        sin = torch.randn(shape[3], device=device, dtype=dtype)

        try:
            torch_out = torch_rotary_emb(q.clone(), k.clone(), cos, sin)
            triton_out = triton_rotary_emb(q.clone(), k.clone(), cos, sin)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: {e}")

if __name__ == "__main__":
    test_rotary_emb()
