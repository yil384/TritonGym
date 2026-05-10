import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.swiglu_fwd import triton_swiglu_fwd
from benchmark.data_v2.torch.swiglu_fwd import torch_swiglu_fwd

def test_swiglu_fwd():
    print("Testing Swiglu forward...")
    input_shapes = [
        ((4, 256),),
        ((8, 512),),
        ((16, 1024),),
        ((32, 2048),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        xy = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_swiglu_fwd(xy)
            triton_out = triton_swiglu_fwd(xy)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"swiglu_fwd case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"swiglu_fwd case {idx}: {e}")

if __name__ == "__main__":
    test_swiglu_fwd()
