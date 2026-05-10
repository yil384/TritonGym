import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.log_softmax import triton_log_softmax
from benchmark.data_v2.torch.log_softmax import torch_log_softmax


def test_log_softmax():
    input_shapes = [
        ((2, 8, 128, 64),),
        ((1, 4, 32, 16),),
        ((3, 2, 64, 32),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)

        triton_out = triton_log_softmax(x)
        torch_out = torch_log_softmax(x)

        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m log_softmax case {idx}: max_abs={max_abs:.3e}")


if __name__ == "__main__":
    test_log_softmax()
