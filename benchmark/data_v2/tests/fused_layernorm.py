import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fused_layernorm import triton_fused_layernorm
from benchmark.data_v2.torch.fused_layernorm import torch_fused_layernorm

def test_fused_layernorm():
    print("Testing fused_layernorm operator...")

    input_shapes = [
        ((4096,), (4096,), (128, 4096)),
        ((2048,), (2048,), (256, 2048)),
        ((8192,), (8192,), (64, 8192)),
        ((1024,), (1024,), (512, 1024)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, (w_shape, b_shape, x_shape) in enumerate(input_shapes):
        weight = torch.randn(*w_shape, dtype=torch.float16, device=device)
        bias = torch.randn(*b_shape, dtype=torch.float16, device=device)
        x = torch.randn(*x_shape, dtype=torch.float16, device=device)

        try:
            torch_out= torch_fused_layernorm(weight, bias, x)
            triton_out= triton_fused_layernorm(weight, bias, x)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m fused_layernorm case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m fused_layernorm case {idx}: {e}")

if __name__ == "__main__":
    test_fused_layernorm()
