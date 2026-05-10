import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.sgmv_expand_slice import triton_sgmv_expand_slice
from benchmark.data_v2.torch.sgmv_expand_slice import torch_sgmv_expand_slice

def test_sgmv_expand_slice():
    print("Testing sgmv_expand_slice operator...")

    input_shapes = [
        ((128, 32), (1, 32, 32)),
        ((64, 16), (1, 16, 16)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for idx, shape in enumerate(input_shapes):
        inputs_shape, lora_b_shape = shape
        rank = lora_b_shape[2]

        inputs = torch.randn(shape[0], dtype=dtype, device=device).contiguous()
        lora_b_weights = torch.randn(shape[1], dtype=dtype, device=device).contiguous()
        
        # Test case 1: add_inputs False
        torch_out = torch_sgmv_expand_slice(
            inputs, lora_b_weights
        )
        triton_out = triton_sgmv_expand_slice(
            inputs, lora_b_weights
        )
        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m shape={shape}, rank={rank}, add_inputs=False, max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_sgmv_expand_slice()
