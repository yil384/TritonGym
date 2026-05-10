import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.layer_norm_fwd import triton_layer_norm_fwd
from benchmark.data_v2.torch.layer_norm_fwd import torch_layer_norm_fwd

def test_layer_norm_fwd():
    print("Testing layer_norm_fwd operator...")

    # Shapes for (batch, feature)
    input_shapes = [
        ((4, 8), (8,), (8,)),
        ((2, 16), (16,), (16,)),
        ((8, 8), (8,), (8,)),
        ((1, 32), (32,), (32,)),
        ((16, 64), (64,), (64,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        weight = torch.randn(shape[1], device=device, dtype=dtype)
        bias = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            # Triton forward
            triton_out = triton_layer_norm_fwd(x, weight, bias, eps=1e-5)

            # Torch forward
            torch_out = torch_layer_norm_fwd(x, weight, bias, eps=1e-5)

            max_abs = (triton_out - torch_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_layer_norm_fwd()
