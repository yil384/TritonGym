import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.quantize_kv_transform import triton_quantize_kv_transform
from benchmark.data_v2.torch.quantize_kv_transform import torch_quantize_kv_transform

def test_quantize_kv_transform():
    print("Testing quantize_kv_transform operator...")

    input_shapes = [
        ((10, 4, 16),),
        ((20, 8, 32),),
        ((5, 2, 8),),
        ((16, 8, 64),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        K = torch.randn(shape[0], device=device, dtype=dtype)

        try:
            # Torch implementation
            torch_out = torch_quantize_kv_transform(K)
            # Triton implementation
            triton_out = triton_quantize_kv_transform(K)

            # Compare results
            max_abs_diff_out = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m quantize_kv_transform case {idx}: input{shape} "
                  f"max_diff_out={max_abs_diff_out:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m quantize_kv_transform case {idx}: input{shape}: {e}")

if __name__ == "__main__":
    test_quantize_kv_transform()
