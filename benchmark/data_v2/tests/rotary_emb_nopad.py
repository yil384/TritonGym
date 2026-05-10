import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rotary_emb_nopad import triton_rotary_emb_nopad
from benchmark.data_v2.torch.rotary_emb_nopad import torch_rotary_emb_nopad

def test_rotary_embedding():
    print("Testing rotary embedding forward...")

    input_shapes = [
        ((32, 8, 64), (32, 4, 64), (64, 64), (64, 64)),  # total_tokens, q_head_num, head_dim
        ((16, 4, 32), (16, 2, 32), (32, 32), (32, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        cos = torch.randn(shape[2], device=device, dtype=dtype)
        sin = torch.randn(shape[3], device=device, dtype=dtype)

        try:
            torch_q = torch_rotary_emb_nopad(q.clone(), k.clone(), cos, sin)
            triton_q = triton_rotary_emb_nopad(q.clone(), k.clone(), cos, sin)
            max_abs_q = (torch_q - triton_q).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: max_abs_q={max_abs_q:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: {e}")

if __name__ == "__main__":
    test_rotary_embedding()
