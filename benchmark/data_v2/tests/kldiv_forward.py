import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.kldiv_forward import triton_kldiv_forward
from benchmark.data_v2.torch.kldiv_forward import torch_kldiv_forward

def test_kldiv():
    print("Testing KL divergence operator...")

    input_shapes = [
        ((2, 3), (2, 3)),
        ((4, 5), (4, 5)),
        ((8, 10), (8, 10)),
        ((16, 16), (16, 16))
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        y_pred = torch.rand(shape[0], device=device, dtype=dtype)
        y_true = torch.rand(shape[1], device=device, dtype=dtype)

        try:
            torch_out = torch_kldiv_forward(y_pred, y_true)
            triton_out = triton_kldiv_forward(y_pred, y_true)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"kldiv case {idx}: shape={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"kldiv case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_kldiv()
