import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fast_rope_embedding import triton_fast_rope_embedding
from benchmark.data_v2.torch.fast_rope_embedding import torch_fast_rope_embedding

def test_fast_rope_embedding():
    print("Testing RoPE embedding operator...")

    input_shapes =[
        ((2, 8, 4, 16), (2, 8, 4, 16), (4, 8), (4, 8)),
        ((1, 4, 8, 32), (1, 4, 8, 32), (8, 16), (8, 16)),
        ((3, 16, 6, 64), (3, 16, 6, 64), (6, 32), (6, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        Q = torch.randn(shape[0], device=device, dtype=dtype)
        K = torch.randn(shape[1], device=device, dtype=dtype)
        cos = torch.randn(shape[2], device=device, dtype=dtype)
        sin = torch.randn(shape[3], device=device, dtype=dtype)

        try:
            torch_out_Q = torch_fast_rope_embedding(Q, K, cos, sin)
            triton_out_Q = triton_fast_rope_embedding(Q, K, cos, sin)

            max_abs_Q = (torch_out_Q - triton_out_Q).abs().max().item()

            print(f"\033[92m OK \033[0m RoPE case {idx}, max_abs_Q={max_abs_Q:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m RoPE case {idx}: {e}")

if __name__ == "__main__":
    test_fast_rope_embedding()
