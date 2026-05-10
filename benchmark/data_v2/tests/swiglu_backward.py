import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.swiglu_backward import triton_swiglu_backward
from benchmark.data_v2.torch.swiglu_backward import torch_swiglu_backward

def test_swiglu_backward():
    print("Testing Swiglu backward...")
    input_shapes = [
        ((4, 256), (4, 128)),  # batch_size=4, hidden=128
        ((8, 128), (8, 64)),
        ((16, 512), (16, 256)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        xy = torch.randn(shape[0], device=device, dtype=dtype)
        dout = torch.randn(shape[1], device=device, dtype=dtype)
        try:
            torch_out = torch_swiglu_backward(xy, dout)
            triton_out = triton_swiglu_backward(xy, dout)
            if isinstance(triton_out, tuple):
                triton_out = torch.cat(triton_out, dim=-1)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"swiglu_backward case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"swiglu_backward case {idx}: {e}")

if __name__ == "__main__":
    test_swiglu_backward()
