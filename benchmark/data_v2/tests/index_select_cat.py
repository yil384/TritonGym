import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.index_select_cat import triton_index_select_cat
from benchmark.data_v2.torch.index_select_cat import torch_index_select_cat

def test_index_select_cat():
    print("Testing index_select_cat operator...")

    input_shapes = [
        ((10, 512), (5, 512)),  # basic test
        ((20, 256), (5, 256)), # larger test
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        source = torch.randn(shape[0], device=device, dtype=dtype)
        output = torch.randn(shape[1], device=device, dtype=dtype)

        try:
            torch_out = torch_index_select_cat(output, source)
            triton_out = triton_index_select_cat(output, source)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"index_select_cat case {idx}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"index_select_cat case {idx}: {e}")

if __name__ == "__main__":
    test_index_select_cat()
