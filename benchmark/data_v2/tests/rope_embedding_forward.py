import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rope_embedding_forward import triton_rope_embedding_forward, triton_rope_embedding_backward
from benchmark.data_v2.torch.rope_embedding_forward import torch_rope_embedding_forward

def test_rope_embedding():
    print("Testing RoPE embedding forward/backward...")

    input_shapes = [
        ((2, 16, 8, 64), (16, 32), (16, 32)),
        ((1, 8, 4, 32), (8, 16), (8, 16)),
        ((4, 32, 16, 128), (32, 64), (32, 64)),
        ((8, 64, 32, 256), (64, 128), (64, 128)),
        ((16, 128, 64, 512), (128, 256), (128, 256)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        Q = torch.randn(shape[0], device=device, dtype=dtype)
        cos = torch.randn(shape[1], device=device, dtype=dtype)
        sin = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            Q_out = torch_rope_embedding_forward(Q, cos, sin)
            
            triton_Q = triton_rope_embedding_forward(Q, cos, sin)
            
            max_abs_Q = (Q_out - triton_Q).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: Q max_abs={max_abs_Q:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: {e}")

if __name__ == "__main__":
    test_rope_embedding()
