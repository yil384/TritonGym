import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.per_block_int8 import triton_per_block_int8
from benchmark.data_v2.torch.per_block_int8 import torch_per_block_int8

def test_per_block_int8():
    print("Testing per_block_int8 operator...")
    # Shapes for (q, k)
    input_shapes = [
        ((2, 256, 64), (2, 256, 64)),
        ((1, 512, 128), (1, 512, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, entry in enumerate(input_shapes):
        q_shape, k_shape = entry
        q = torch.randn(*q_shape, device=device, dtype=dtype)
        k = torch.randn(*k_shape, device=device, dtype=dtype)
        try:
            torch_out = torch_per_block_int8(q, k)
            triton_out = triton_per_block_int8(q, k)
            # Compare q_int8 and k_int8
            max_abs_q = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m per_block_int8 case {idx}: max_abs_q={max_abs_q:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m per_block_int8 case {idx}: {e}")

if __name__ == "__main__":
    test_per_block_int8()
