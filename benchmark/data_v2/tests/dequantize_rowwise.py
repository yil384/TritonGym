import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.dequantize_rowwise import triton_dequantize_rowwise
from benchmark.data_v2.torch.dequantize_rowwise import torch_dequantize_rowwise

def test_dequantize_rowwise():
    print("Testing dequantize_rowwise...")

    input_shapes = [
        ((2, 4), (2,)),       # small
        ((10, 16), (10,)),    # medium
        ((5, 8), (5,)),       # zeros case
        ((3, 32), (3,)),      # larger block
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_state = torch.float32

    for idx, (x_shape, state_shape) in enumerate(input_shapes):
        x = torch.randn(x_shape, dtype=dtype_state, device=device)
        state_x = torch.rand(state_shape, dtype=dtype_state, device=device)

        try:
            torch_out = torch_dequantize_rowwise(x, state_x)
            triton_out = triton_dequantize_rowwise(x, state_x)
            max_abs = (torch_out- triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: x{x_shape}, state_x{state_shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: x{x_shape}, state_x{state_shape}: {e}")

if __name__ == "__main__":
    test_dequantize_rowwise()
