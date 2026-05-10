import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.relu_strided_buffer import triton_relu_strided_buffer
from benchmark.data_v2.torch.relu_strided_buffer import torch_relu_strided_buffer

def test_relu_strided_buffer():
    print("Testing relu_strided_buffer operator...")

    input_shapes = [
        ((512,), (512,)),   # both inputs same size
        ((100,), (100,)),   # small vector
        ((1025,), (1025,)), # non-multiple of block size
        ((4096,), (4096,)), # large vector
        ((10000,), (10000,)), # very large
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        b = torch.randn(shape[1], device=device, dtype=dtype)
        try:
            torch_out = torch_relu_strided_buffer(a, b)
            triton_out = triton_relu_strided_buffer(a, b)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"relu_strided_buffer case {idx}: input{shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"relu_strided_buffer case {idx}: input{shape}: {e}")

if __name__ == "__main__":
    test_relu_strided_buffer()
