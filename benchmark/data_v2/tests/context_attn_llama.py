import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.context_attn_llama import triton_context_attn_llama
from benchmark.data_v2.torch.context_attn_llama import torch_context_attn_llama

def test_context_attn():
    print("Testing context attention operator...")

    input_shapes = [
        ((16, 16, 128), (32, 16, 128), (32, 16, 128)),
        ((32, 8, 64), (48, 8, 64), (48, 8, 64)),
        ((8, 32, 256), (24, 32, 256), (24, 32, 256)),
        ((4, 4, 16), (20, 4, 16), (20, 4, 16)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)
        try:
            out_torch = torch_context_attn_llama(q, k, v)
            out_triton = triton_context_attn_llama(q, k, v)

            max_abs = (out_torch - out_triton).abs().max().item()
            print(f"\033[92m OK \033[0m context_attn case {idx}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m context_attn case {idx}: {e}")

if __name__ == "__main__":
    test_context_attn()
