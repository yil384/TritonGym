import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fused_recurrent_rwkv6 import triton_fused_recurrent_rwkv6
from benchmark.data_v2.torch.fused_recurrent_rwkv6 import torch_fused_recurrent_rwkv6

def test_fused_recurrent_rwkv6():
    print("Testing fused_recurrent_rwkv6 operator...")

    input_shapes = [
        ((2, 2, 4, 8), (2, 2, 4, 8), (2, 2, 4, 8), (2, 2, 4, 8), (2, 8)),  # case 0: small test
        ((1, 4, 6, 16), (1, 4, 6, 16), (1, 4, 6, 16), (1, 4, 6, 16), (4, 16)),  # case 1: medium test
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        r = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)
        w = torch.randn(shape[3], device=device, dtype=dtype)
        u = torch.randn(shape[4], device=device, dtype=dtype)

        try:
            torch_out = torch_fused_recurrent_rwkv6(r, k, v, w, u)
            triton_out = triton_fused_recurrent_rwkv6(r, k, v, w, u)
            max_abs_out = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m fused_recurrent_rwkv6 case {idx}: max_abs_out={max_abs_out:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m fused_recurrent_rwkv6 case {idx}: {e}")


if __name__ == "__main__":
    test_fused_recurrent_rwkv6()
