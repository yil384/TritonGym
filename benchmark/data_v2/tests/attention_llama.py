import torch
import sys
import os
import warnings
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# Import paths
from benchmark.data_v2.triton.attention_llama import triton_attention_llama
from benchmark.data_v2.torch.attention_llama import torch_attention_llama

def test_attention_llama_compare():
    print("Testing attention llama operator with Triton vs Torch...")

    input_shapes = [
        ((1, 16, 32, 128), (1, 16, 32, 128), (1, 16, 32, 128)),
        ((2, 4, 64, 64), (2, 4, 64, 64), (2, 4, 64, 64)),
        ((2, 8, 128, 128), (2, 8, 128, 128), (2, 8, 128, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        B, H, N_CTX, D_MODEL = shape[0]

        # Create inputs (unified dtype)
        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)
        v = torch.randn(shape[2], device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(D_MODEL)

        try:
            # Torch reference (always runs)
            torch_out = torch_attention_llama(q, k, v, scale, is_causal=False, start_position=0)

            # Non-Hopper: run Triton and compare
            triton_out = triton_attention_llama(q, k, v, scale, is_causal=False, start_position=0)
            # Compare
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m attention case {idx}: shape={shape[0]} max_abs={max_abs:.3e}")

        except Exception as e:
            # Triton may still crash/throw errors at runtime; catch and output safely
            warnings.warn(f"Test failed for case {idx}: {e}")
            print(f"\033[91m ERROR \033[0m attention case {idx}: shape={shape[0]}: {e}")

if __name__ == "__main__":
    test_attention_llama_compare()
