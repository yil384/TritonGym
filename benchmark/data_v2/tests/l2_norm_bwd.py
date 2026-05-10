import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.l2_norm_bwd import triton_l2_norm_bwd
from benchmark.data_v2.torch.l2_norm_bwd import torch_l2_norm_bwd

def test_l2_norm_bwd():
    print("Testing l2_norm_bwd operator...")

    # Shapes for (batch, feature_dim)
    input_shapes = [
        ((4, 8), (4, 8)),
        ((2, 16), (2, 16)),
        ((8, 8), (8, 8)),
        ((1, 8), (1, 8)),
        ((16, 32), (16, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        dy = torch.randn(shape[1], device=device, dtype=dtype)

        try:
            torch_out = torch_l2_norm_bwd(x, dy)
            triton_out = triton_l2_norm_bwd(x, dy)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_l2_norm_bwd()
