import torch
import sys
import os

# Ensure the benchmark package can be found
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.add_value import triton_add_value
from benchmark.data_v2.torch.add_value import torch_add_value

def test_puzzle1():
    print("Testing puzzle1 (add constant 10) operator...")
    # Test different input sizes
    input_shapes = [
        ((4,),),        # small array
        ((8,),),        # medium array
        ((1024,),),     # large array
        ((4096,),),     # larger array
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(*shape, device=device, dtype=dtype)
        try:
            torch_out = torch_add_value(a)
            triton_out = triton_add_value(a)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"puzzle1 case {idx}: shape={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"puzzle1 case {idx}: shape={shape}: {e}")


if __name__ == "__main__":
    test_puzzle1()
