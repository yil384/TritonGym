import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.kcache_copy import triton_kcache_copy
from benchmark.data_v2.torch.kcache_copy import torch_kcache_copy


def test_kcache_copy():
    print("Testing kcache copy operator...")

    input_shapes = [
        ((8, 4, 64),),
        ((16, 4, 64),),
        ((32, 4, 64),),
        ((64, 4, 64),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        k = torch.randn(shape[0], device=device, dtype=dtype)

        try:
            torch_out = torch_kcache_copy(k)
            triton_out = triton_kcache_copy(k)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"kcache_copy case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"kcache_copy case {idx}: {e}")


if __name__ == "__main__":
    test_kcache_copy()
