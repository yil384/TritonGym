import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.cache_transform import triton_cache_transform
from benchmark.data_v2.torch.cache_transform import torch_cache_transform  

def test_cache_transform():
    print("Testing cache_transform operator...")

    # Shapes: (num_seqs, max_length, hidden_dim)
    input_shapes = [
        # (num_seqs, max_length, hidden_dim)
        (8, 20, 64),
        (4, 16, 128),
        (16, 32, 256),
    ]
    input_shapes = [
        ((20, 64), (20, 64)),
        ((16, 128), (16, 128)),
        ((32, 256), (32, 256)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        cos_cache = torch.randn(shape[0], dtype=dtype, device=device)
        sin_cache = torch.randn(shape[1], dtype=dtype, device=device)

        try:
            # Triton output
            cos_triton = triton_cache_transform(cos_cache, sin_cache)
            # Torch output
            cos_torch = torch_cache_transform(cos_cache, sin_cache)
            
            max_abs_cos = (cos_triton - cos_torch).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"cache_transform case {idx}: num_seqs={cos_cache.shape[0]}, max_length={cos_cache.shape[0]}, hidden_dim={cos_cache.shape[1]}, max_abs_cos={max_abs_cos:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"cache_transform case {idx}: num_seqs={cos_cache.shape[0]}, max_length={cos_cache.shape[0]}, hidden_dim={cos_cache.shape[1]}: {e}")

if __name__ == "__main__":
    test_cache_transform()
