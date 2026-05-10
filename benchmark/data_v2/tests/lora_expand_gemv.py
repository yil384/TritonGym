import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.lora_expand_gemv import triton_lora_expand_gemv
from benchmark.data_v2.torch.lora_expand_gemv import torch_lora_expand_gemv


def test_lora_expand():
    input_shapes = [
        ((4, 128), (3, 64, 128)),  # batch_size, hidden_size ; lora_num, rank, hidden_size
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Create inputs
    inputs = torch.randn(input_shapes[0][0], device=device, dtype=dtype)
    lora_b_weights = torch.randn(input_shapes[0][1], device=device, dtype=dtype)

    # Case 1: add_inputs=True
    output_triton = triton_lora_expand_gemv(
        inputs, lora_b_weights
    )
    output_torch = torch_lora_expand_gemv(
        inputs, lora_b_weights,
    )
    max_abs = (output_torch - output_triton).abs().max().item()
    print(f"\033[92m OK \033[0m lora_expand case 1 max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_lora_expand()
