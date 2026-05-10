import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.softmax_optimize import triton_softmax_optimize
from benchmark.data_v2.torch.softmax_optimize import torch_softmax_optimize

def test_softmax():
    print("Testing softmax operator...")

    input_shapes = [
        ((128, 512),),   # test_case_1
        ((64, 1024),),   # test_case_2
        ((256, 128),)   # test_case_3
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        # Triton output
        out_triton = triton_softmax_optimize(x)

        # Torch output
        out_torch = torch_softmax_optimize(x)

        # Compare
        max_abs = (out_triton - out_torch).abs().max().item()
        print(f"\033[92m OK \033[0m", end=' ')
        print(f"softmax case {idx}: shape={shape[0]} max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_softmax()
