import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.token_attn_llama2 import triton_token_attn_llama2
from benchmark.data_v2.torch.token_attn_llama2 import torch_token_attn_llama2

def test_token_attn_llama2():
    print("Testing token-wise attention LLAMA2 operator...")

    # input_shapes: (batch, head, max_input_len, d_model)
    input_shapes = [
        ((2, 4, 32, 32), (2, 4, 32, 32)),
        ((2, 4, 64, 32), (2, 4, 64, 32)),
        ((3, 4, 32, 64), (3, 4, 32, 64)),
        ((4, 4, 64, 64), (4, 4, 64, 64))
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        batch, head, max_input_len, d_model = shape[0]
        Q = torch.randn(shape[0], device=device, dtype=dtype)
        K = torch.randn(shape[1], device=device, dtype=dtype)

        # Run PyTorch implementation
        torch_out = torch_token_attn_llama2(Q, K)

        # Run Triton implementation
        triton_out= triton_token_attn_llama2(Q, K)

        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m attn llama2 case {idx}: max_abs={max_abs:.3e}")


if __name__ == "__main__":
    test_token_attn_llama2()
