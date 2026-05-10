import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rms_norm import triton_rms_norm
from benchmark.data_v2.torch.rms_norm import torch_rms_norm

def test_rms_norm():
    print("Testing RMS normalization...")

    input_shapes = [
        ((32, 128), (128,)),
        ((64, 256), (256,)),
        ((16, 1024), (1024,)),
        ((8, 512), (512,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        weight = torch.randn(shape[1], device=device, dtype=dtype)

        try:
            torch_out = torch_rms_norm(x, weight)
            triton_out = triton_rms_norm(x, weight)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: shape={shape}: {e}")


if __name__ == "__main__":
    test_rms_norm()
