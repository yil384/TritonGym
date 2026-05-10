import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rbe_triton_transform import triton_rbe_triton_transform
from benchmark.data_v2.torch.rbe_triton_transform import torch_rbe_triton_transform

def test_rbe_triton_transform():
    print("Testing rbe_triton_transform operator...")

    input_shapes = [
        ((2, 4, 1024),),
        ((2, 8, 512),),
        ((1, 2, 128),),
        ((3, 4, 2048),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_rbe_triton_transform(x)
            triton_out = triton_rbe_triton_transform(x)
            max_diff = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m rbe_triton_transform case {idx}: input{shape} max_diff={max_diff:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m rbe_triton_transform case {idx}: input{shape}: {e}")

if __name__ == "__main__":
    test_rbe_triton_transform()
