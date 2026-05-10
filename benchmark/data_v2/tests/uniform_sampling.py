import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.uniform_sampling import triton_uniform_sampling
from benchmark.data_v2.torch.uniform_sampling import torch_uniform_sampling

def test_uniform():
    print("Testing uniform sampling operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # input_shapes: list of tuples (tensor shape, from_, to_)
    input_shapes = [
        ((1000,),),
    ]

    for idx, shape in enumerate(input_shapes):
        # Triton tensor
        triton_tensor = torch.randn(shape[0], device=device, dtype=dtype)
        triton_uniform_sampling(triton_tensor)

        # Torch tensor
        torch_tensor = triton_tensor.clone()
        torch_uniform_sampling(torch_tensor)

        # Compare min/max
        max_abs_diff = (torch_tensor - triton_tensor).abs().max().item()
        print(f"\033[92m OK \033[0m uniform case {idx}: shape={shape}, max_abs_diff={max_abs_diff:.3e}")

if __name__ == "__main__":
    test_uniform()
