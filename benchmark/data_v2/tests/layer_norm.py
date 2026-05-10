import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.layer_norm import triton_layer_norm
from benchmark.data_v2.torch.layer_norm import torch_layer_norm

def test_layer_norm():
    print("Testing layer_norm operator...")
    # Shapes for input
    input_shapes = [
        ((4, 8),),
        ((2, 16),),
        ((8, 8),),
        ((1, 32),),
        ((16, 64),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        try:
            torch_out = torch_layer_norm(x)
            triton_out = triton_layer_norm(x)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx} forward: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_layer_norm()
