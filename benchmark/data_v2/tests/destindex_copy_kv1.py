import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.destindex_copy_kv1 import triton_destindex_copy_kv1
from benchmark.data_v2.torch.destindex_copy_kv1 import torch_destindex_copy_kv1

def test_destindex_copy_kv():
    print("Testing destindex_copy_kv...")

    input_shapes = [
        ((32, 12, 96),),
        ((16, 8, 64),),
        ((4, 2, 16),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        src = torch.randn(shape[0], device=device, dtype=dtype)
        
        try:
            torch_out = torch_destindex_copy_kv1(src)
            triton_out = triton_destindex_copy_kv1(src)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_destindex_copy_kv()
