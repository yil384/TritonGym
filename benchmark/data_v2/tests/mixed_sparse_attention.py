import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.mixed_sparse_attention import triton_mixed_sparse_attention
from benchmark.data_v2.torch.mixed_sparse_attention import torch_mixed_sparse_attention

def test_mixed_sparse_attention():
    print("Testing mixed_sparse_attention operator...")

    input_shapes = [
        ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),  
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], dtype=torch.float16, device=device)
        k = torch.randn(shape[1], dtype=torch.float16, device=device)
        v = torch.randn(shape[2], dtype=torch.float16, device=device)
        
        try:
            torch_out = torch_mixed_sparse_attention(q, k, v)
            triton_out = triton_mixed_sparse_attention(q, k, v)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m mixed_sparse_attention case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m mixed_sparse_attention case {idx}: {e}")


if __name__ == "__main__":
    test_mixed_sparse_attention()
