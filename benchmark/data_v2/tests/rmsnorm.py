import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rmsnorm import triton_rmsnorm
from benchmark.data_v2.torch.rmsnorm import torch_rmsnorm

def test_rmsnorm_triton_func():
    print("Testing RMSNorm Triton -> Torch implementation...")

    input_shapes = [
        ((2, 4, 1024), (1024,)),  # batch, M, K
        ((2, 4, 1024), (1024,)),
        ((3, 4, 1024), (1024,)),
        ((2, 5, 1024), (1024,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        rms_w = torch.randn(shape[1], device=device, dtype=dtype)
        try:
            torch_out = torch_rmsnorm(x, rms_w)
            triton_out = triton_rmsnorm(x, rms_w)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: x{shape[0]}, rms_w{shape[1]}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: x{shape[0]}, rms_w{shape[1]}: {e}")


if __name__ == "__main__":
    test_rmsnorm_triton_func()
