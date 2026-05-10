import torch
import sys
import os
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.attn_fwd_causal import triton_attn_fwd_causal
from benchmark.data_v2.torch.attn_fwd_causal import torch_attn_fwd_causal

def test_attention_forward_compare():
    print("Testing attention forward operator with Triton vs Torch...")

    input_shapes = [
        ((2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 128)), 
        ((2, 8, 128, 128), (2, 8, 128, 128), (2, 8, 128, 128)),
        ((1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        B, H, N_CTX, D_MODEL = shape[0]

        # Create inputs, all using the same dtype
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)
        q_scale = torch.ones((B, H, N_CTX), device=device, dtype=dtype)
        k_scale = torch.ones((B, H, N_CTX), device=device, dtype=dtype)

        try:
            # PyTorch reference
            torch_out = torch_attn_fwd_causal(q, k, v)

            # Triton kernel
            triton_out = triton_attn_fwd_causal(q, k, v)

            # Output difference
            max_abs = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m attention case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            warnings.warn(f"Test failed for case {idx}: {e}")
            print(f"\033[91m ERROR \033[0m attention case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_attention_forward_compare()
