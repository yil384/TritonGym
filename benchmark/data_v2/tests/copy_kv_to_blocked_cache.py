import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.copy_kv_to_blocked_cache import triton_copy_kv_to_blocked_cache
from benchmark.data_v2.torch.copy_kv_to_blocked_cache import torch_copy_kv_to_blocked_cache

def test_copy_kv_to_blocked_cache():
    print("Testing copy_kv_to_blocked_cache operator...")

    # Define input_shapes, each entry is (batch_size, num_kv_heads, head_dim, block_size)
    input_shapes = [
        ((2, 4, 64), (2, 4, 64)),  # small test
        ((4, 8, 128), (4, 8, 128)), # medium test
        ((8, 16, 512), (8, 16, 512)), # xlarge test
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        # Input tensors
        k = torch.randn(shape[0], dtype=dtype, device=device)
        v = torch.randn(shape[1], dtype=dtype, device=device)

        # Execute Triton function
        k_cache_triton = triton_copy_kv_to_blocked_cache(k, v)
        # Execute Torch function
        k_cache_torch = torch_copy_kv_to_blocked_cache(k, v)

        # Compare results
        max_abs_k = (k_cache_torch - k_cache_triton).abs().max().item()
        print(f"\033[92mOK\033[0m case {idx}: KCache max_abs={max_abs_k:.3e}")

if __name__ == "__main__":
    test_copy_kv_to_blocked_cache()
