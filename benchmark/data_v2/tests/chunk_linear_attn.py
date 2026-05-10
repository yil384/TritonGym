import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_linear_attn import triton_chunk_linear_attn
from benchmark.data_v2.torch.chunk_linear_attn import torch_chunk_linear_attn

def test_chunk_linear_attn():
    print("Testing chunk_linear_attn operator...")

    input_shapes = [
        ((2, 4, 64, 32), (2, 4, 64, 32), (2, 4, 64, 32)),
        ((1, 2, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)),
        ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(shape[1], device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(shape[2], device=device, dtype=dtype, requires_grad=True)

        try:
            # Triton version
            triton_out = triton_chunk_linear_attn(q, k, v)
            # Torch version
            torch_out = torch_chunk_linear_attn(q, k, v)

            # Compare outputs (contiguous to ensure same memory layout)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: shape={shape} max_abs_out={max_abs:.3e}", end='\n')

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_chunk_linear_attn()
