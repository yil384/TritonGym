import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.conv2d_fwd import triton_conv2d_fwd
from benchmark.data_v2.torch.conv2d_fwd import torch_conv2d_fwd

def test_conv2d():
    print("Testing conv2d operator...")
    input_shapes = [
        ((1, 3, 32, 32), (16, 3, 3, 3)),  # simple
        ((2, 3, 64, 64), (32, 3, 5, 5)),  # medium
        ((4, 3, 128, 128), (64, 3, 7, 7)), # large
        ((8, 3, 256, 256), (128, 3, 3, 3)), # xlarge
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, (in_shape, weight_shape) in enumerate(input_shapes):
        input_tensor = torch.randn(*in_shape, device=device, dtype=dtype)
        weight_tensor = torch.randn(*weight_shape, device=device, dtype=dtype)

        try:
            torch_out = torch_conv2d_fwd(input_tensor, weight_tensor)
            triton_out = triton_conv2d_fwd(input_tensor, weight_tensor)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m conv2d case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m conv2d case {idx}: {e}")


if __name__ == "__main__":
    test_conv2d()
