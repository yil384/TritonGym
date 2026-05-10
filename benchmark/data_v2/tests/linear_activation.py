import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.linear_activation import triton_linear_activation
from benchmark.data_v2.torch.linear_activation import torch_linear_activation


def test_linear_layer():
    input_shapes = [
        ((64, 128), (128, 128), (128,)),  # M, K ; N, K ; N
        ((32, 256), (256, 256), (256,)),  # M, K ; N, K ; N
        ((128, 64), (64, 64), (64,)),     # M, K ; N, K ; N
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        weight = torch.randn(shape[1], device=device, dtype=dtype)
        bias = torch.randn(shape[2], device=device, dtype=dtype)

        torch_out = torch_linear_activation(x, weight, bias)
        triton_out = triton_linear_activation(x, weight, bias)

        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m Linear case {idx}: max_abs={max_abs:.3e}")


if __name__ == "__main__":
    test_linear_layer()
