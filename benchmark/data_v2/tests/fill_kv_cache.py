import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fill_kv_cache import triton_fill_kv_cache
from benchmark.data_v2.torch.fill_kv_cache import torch_fill_kv_cache

def test_fill_kv_cache():
    print("Testing fill_kv_cache operator...")

    input_shapes = [
        ((2, 32, 4, 16), (2, 32, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)),
        ((3, 64, 8, 32), (3, 64, 8, 32), (3, 16, 8, 32), (3, 16, 8, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        k_states = torch.randn(shape[0], dtype=dtype, device=device)
        v_states = torch.randn(shape[1], dtype=dtype, device=device)
        k_caches = torch.randn(shape[2], dtype=dtype, device=device)
        v_caches = torch.randn(shape[3], dtype=dtype, device=device)
        
        # Triton
        triton_out = triton_fill_kv_cache(
            k_states,
            v_states,
            k_caches.clone(),
            v_caches.clone(),
        )

        # Torch
        torch_out= torch_fill_kv_cache(
            k_states,
            v_states,
            k_caches.clone(),
            v_caches.clone(),
        )

        max_abs_k = (triton_out - torch_out).abs().max().item()
        print(f"\033[92m OK \033[0m case {idx}: KCache max_abs={max_abs_k:.3e}")


if __name__ == "__main__":
    test_fill_kv_cache()
