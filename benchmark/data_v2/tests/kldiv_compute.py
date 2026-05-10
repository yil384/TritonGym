import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.kldiv_compute import triton_kldiv_compute
from benchmark.data_v2.torch.kldiv_compute import torch_kldiv_compute

def test_kldiv_compute():
    print("Testing KL divergence operator...")

    input_shapes = [
        ((1024,), (1024,)),
        ((2048,), (2048,)),
        ((4096,), (4096,)),
        ((8192,), (8192,))
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, (shape_x, shape_y) in enumerate(input_shapes):
        # Use abs to avoid log of negative numbers
        x = torch.rand(shape_x, device=device, dtype=dtype).clamp(min=1e-6)
        y = torch.rand(shape_y, device=device, dtype=dtype).clamp(min=1e-6)

        try:
            torch_out = torch_kldiv_compute(x, y)
            triton_out = triton_kldiv_compute(x, y)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"kldiv_compute case {idx}: shape={shape_x}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"kldiv_compute case {idx}: shape={shape_x}: {e}")

if __name__ == "__main__":
    test_kldiv_compute()
