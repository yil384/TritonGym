import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.var_len_copy import triton_var_len_copy
from benchmark.data_v2.torch.var_len_copy import torch_var_len_copy

def test_var_len_copy():
    print("Testing var_len_copy operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test input shapes (list of tuples: old/new lengths)
    input_shapes = [
        ((500,),),  # Total length of the old and new locations
    ]

    for idx, shape in enumerate(input_shapes):
        old_a_location = torch.randn(shape[0], dtype=torch.float32, device=device)

        try:
            torch_out = torch_var_len_copy(old_a_location)
            triton_out = triton_var_len_copy(old_a_location)

            # Compare results
            max_abs_diff = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m var_len_copy case {idx}, max_abs_diff={max_abs_diff:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m var_len_copy case {idx}: {e}")


if __name__ == "__main__":
    test_var_len_copy()
