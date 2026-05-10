import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.dropout import triton_dropout
from benchmark.data_v2.torch.dropout import torch_dropout

def test_dropout():
    print("Testing dropout operator...")

    input_shapes = [
        ((10,), (10,)),
        ((32, 64), (32, 64)),
        ((5, 3, 7), (5, 3, 7)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        x_keep = torch.randn(shape[1], device=device, dtype=dtype)
        try:
            torch_out = torch_dropout(x, x_keep)
            triton_out = triton_dropout(x.clone(), x_keep.clone())
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m dropout case {idx} shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m dropout case {idx} shape={shape}: {e}")


if __name__ == "__main__":
    test_dropout()
