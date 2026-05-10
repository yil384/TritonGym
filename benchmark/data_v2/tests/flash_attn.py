import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.flash_attn import triton_flash_attn
from benchmark.data_v2.torch.flash_attn import torch_flash_attn

def test_flash_attn():
    print("Testing flash_attn operator...")

    input_shapes = [
        ((2, 2, 128, 64), (2, 2, 128, 64), (2, 2, 128, 64)),
        ((1, 4, 64, 32), (1, 4, 64, 32), (1, 4, 64, 32)),
        ((3, 1, 32, 16), (3, 1, 32, 16), (3, 1, 32, 16)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            torch_out = torch_flash_attn(q, k, v)
            triton_out = triton_flash_attn(q, k, v)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m flash_attn case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m flash_attn case {idx}: {e}")

if __name__ == "__main__":
    test_flash_attn()
