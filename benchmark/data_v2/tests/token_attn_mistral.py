import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.token_attn_mistral import triton_token_attn_mistral
from benchmark.data_v2.torch.token_attn_mistral import torch_token_attn_mistral

def test_token_attn_mistral():
    print("Testing token-wise attention Mistral operator...")

    input_shapes = [
        ((4, 128), (2, 128, 64)),
        ((4, 64), (2, 64, 64)),
        ((4, 256), (2, 256, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        Prob = torch.randn(shape[0], dtype=dtype, device=device)
        V = torch.randn(shape[1], dtype=dtype, device=device)

        # run both implementations
        torch_out = torch_token_attn_mistral(Prob, V)
        triton_out = triton_token_attn_mistral(Prob, V)

        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m attn mistral case {idx}: max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_token_attn_mistral()
