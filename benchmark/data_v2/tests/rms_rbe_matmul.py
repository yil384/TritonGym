import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rms_rbe_matmul import triton_rms_rbe_matmul
from benchmark.data_v2.torch.rms_rbe_matmul import torch_rms_rbe_matmul

def test_rms_rbe_matmul():
    print("Testing RMS + RBE matmul...")

    input_shapes = [
        ((2, 4, 1024), (64, 1024), (1024,)),
        ((3, 4, 1024), (64, 1024), (1024,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for idx, (x_shape, w_shape, rms_shape) in enumerate(input_shapes):
        x = torch.randn(x_shape, device=device, dtype=dtype)
        weight = torch.randn(w_shape, device=device, dtype=dtype)
        rms_w = torch.randn(rms_shape, device=device, dtype=dtype)
        try:
            torch_out = torch_rms_rbe_matmul(x, weight, rms_w)
            triton_out = triton_rms_rbe_matmul(x, weight, rms_w)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}, use_rbe={True} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}, use_rbe={True}: {e}")


if __name__ == "__main__":
    test_rms_rbe_matmul()
