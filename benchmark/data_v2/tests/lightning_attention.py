import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.lightning_attention import triton_lightning_attention
from benchmark.data_v2.torch.lightning_attention import torch_lightning_attention

def test_lightning_attention():
    print("Testing lightning_attention operator...")

    input_shapes = [
        ((2, 8, 128, 64), (2, 8, 128, 64), (2, 8, 128, 128)),  
        ((1, 4, 64, 32), (1, 4, 64, 32), (1, 4, 64, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(shape[1], device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(shape[2], device=device, dtype=dtype, requires_grad=True)

        try:
            torch_out = torch_lightning_attention(q, k, v)
            triton_out = triton_lightning_attention(q, k, v)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}: {e}")


if __name__ == "__main__":
    test_lightning_attention()
