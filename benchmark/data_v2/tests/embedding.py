import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.embedding import triton_embedding
from benchmark.data_v2.torch.embedding import torch_embedding

def test_embedding():
    print("Testing embedding operator...")

    input_shapes = [
        ((1000, 512),),        
        ((500, 512),),
        ((200, 512),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        weight = torch.randn(shape[0], dtype=dtype, device=device)

        try:
            torch_out = torch_embedding(weight)
            out_triton = triton_embedding(weight)
            max_abs = (torch_out - out_triton).abs().max().item()
            print(f"\033[92m OK \033[0m embedding case {idx}: seq_len={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m embedding case {idx}: seq_len={shape}: {e}")

if __name__ == "__main__":
    test_embedding()
