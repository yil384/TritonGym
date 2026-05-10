import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.parallel_attention import triton_parallel_attention
from benchmark.data_v2.torch.parallel_attention import torch_parallel_attention

def test_parallel_rebased():
    print("Testing parallel_rebased attention...")

    input_shapes = [
        ((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16)),
        ((1, 2, 16, 32), (1, 2, 16, 32), (1, 2, 16, 32)),
        ((2, 2, 32, 64), (2, 2, 32, 64), (2, 2, 32, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)

        torch_out = torch_parallel_attention(q, k, v)
        triton_out = triton_parallel_attention(q, k, v)

        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m parallel_rebased case {idx}: max_abs={max_abs:.3e}")


if __name__ == "__main__":
    test_parallel_rebased()
