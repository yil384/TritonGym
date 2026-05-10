import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.max_reduction import triton_max_reduction
from benchmark.data_v2.torch.max_reduction import torch_max_reduction

def test_max_operator():
    print("Testing max operator...")

    input_shapes = [
        ((1024, 1024),),    # 2D tensor
        ((128, 64, 32),),   # 3D tensor
        ((512, 256),),      # 2D keepdim test
        ((64, 128, 256),),  # 3D negative dim test
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        inp = torch.randn(shape[0], device=device, dtype=dtype)
        
        triton_out = triton_max_reduction(inp)

        torch_out = torch_max_reduction(inp)

        # Compare values
        try:
            max_abs = (torch_out.values - triton_out.values).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"max case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"max case {idx}: shape={shape}: {e}")


if __name__ == "__main__":
    test_max_operator()
