import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.layernorm_fwd import triton_layernorm_fwd
from benchmark.data_v2.torch.layernorm_fwd import torch_layernorm_fwd

def test_layernorm_fwd():
    print("Testing layernorm_fwd operator...")

    # Input shapes to test
    input_shapes = [
        ((2, 3, 128), (3, 128)),  # (B, H, F), (H, F)
        ((4, 3, 128), (3, 128)),
        ((2, 3, 256), (3, 256)),
        ((2, 4, 128), (4, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        X = torch.randn(shape[0], dtype=dtype, device=device)
        W = torch.randn(shape[1], dtype=dtype, device=device)
        try:
            torch_out = torch_layernorm_fwd(X, W)
            triton_out = triton_layernorm_fwd(X, W)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}: {e}")


if __name__ == "__main__":
    test_layernorm_fwd()
