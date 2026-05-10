import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunked_cumsum_fwd import triton_chunked_cumsum_fwd
from benchmark.data_v2.torch.chunked_cumsum_fwd import torch_chunked_cumsum_fwd


def test_chunked_cumsum_fwd():
    print("Testing chunked_cumsum_fwd operator...")

    input_shapes = [
        ((2, 10, 4), (4,), (4,)),
        ((1, 16, 8), (8,), (8,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        dt = torch.randn(shape[0], device=device, dtype=dtype)
        A = torch.randn(shape[1], device=device, dtype=dtype)
        dt_bias = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            torch_out = torch_chunked_cumsum_fwd(dt, A, dt_bias)
            triton_out = triton_chunked_cumsum_fwd(dt, A, dt_bias)

            max_abs1 = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=" ")
            print(f"case {idx}: max_abs_cumsum={max_abs1:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=" ")
            print(f"case {idx}: {e}")


if __name__ == "__main__":
    test_chunked_cumsum_fwd()
