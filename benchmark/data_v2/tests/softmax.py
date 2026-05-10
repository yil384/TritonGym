import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.softmax import triton_softmax
from benchmark.data_v2.torch.softmax import torch_softmax

def test_softmax():
    print("Testing softmax operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Input shapes to test
    input_shapes = [
        ((2, 2),),
        ((3, 3),),
        ((4, 4),),
        ((1, 5),),
        ((5, 1),),
        ((128, 512),),
        ((64, 1024),),
    ]

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        try:
            torch_out = torch_softmax(x)
            triton_out = triton_softmax(x)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m softmax case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m softmax case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_softmax()
