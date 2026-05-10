# benchmark/data/tests/cross_entropy_test.py
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.cross_entropy_loss import triton_cross_entropy_loss
from benchmark.data_v2.torch.cross_entropy_loss import torch_cross_entropy_loss

def test_cross_entropy():
    print("Testing cross-entropy function...")

    input_shapes = [
        ((4, 10),),
        ((8, 20),),
        ((16, 50),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        logits = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_cross_entropy_loss(logits)
            triton_out = triton_cross_entropy_loss(logits)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}, {e}")

if __name__ == "__main__":
    test_cross_entropy()
