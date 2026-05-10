import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.parallel_retention_attention import triton_parallel_retention_attention
from benchmark.data_v2.torch.parallel_retention_attention import torch_parallel_retention_attention

def test_parallel_retention_attention():
    print("Testing parallel retention attention operator...")

    # Test input shape list
    input_shapes = [
        # (q_shape, k_shape, v_shape)
        ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
        ((2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 64)),
        ((4, 4, 128, 64), (4, 4, 128, 64), (4, 4, 128, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, (q_shape, k_shape, v_shape) in enumerate(input_shapes):
        q = torch.randn(*q_shape, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(*k_shape, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(*v_shape, device=device, dtype=dtype, requires_grad=True)
        try:
            torch_out = torch_parallel_retention_attention(q, k, v)
            triton_out = triton_parallel_retention_attention(q, k, v)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: {e}")

if __name__ == "__main__":
    test_parallel_retention_attention()
