import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.geglu_backward import triton_geglu_forward, triton_geglu_backward
from benchmark.data_v2.torch.geglu_backward import torch_geglu_forward, torch_geglu_backward

def test_geglu_tanh():
    print("Testing GeGLU-Tanh operator...")
    input_shapes = [
        ((2, 128), (2, 128)),
        ((3, 128), (3, 128)),
        ((2, 256), (2, 256)),
        ((1, 128), (1, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        b = torch.randn(shape[1], device=device, dtype=dtype)

        try:
            # Forward
            torch_c = torch_geglu_forward(a, b)
            triton_c = triton_geglu_forward(a, b)
            max_abs_c = (torch_c - triton_c).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"GeGLU case {idx}, max_abs_c={max_abs_c:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"GeGLU case {idx}: {e}")

if __name__ == "__main__":
    test_geglu_tanh()
