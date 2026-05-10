import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rmsnorm_fused_llama import triton_rmsnorm_fused_llama
from benchmark.data_v2.torch.rmsnorm_fused_llama import torch_rmsnorm_fused_llama

def test_rmsnorm_fused_llama_func():
    print("Testing RMSNorm fused LLaMA implementation...")

    input_shapes = [
        ((2, 64), (64,)),
        ((4, 128), (128,)),
        ((1, 8192), (8192,)),
        ((1, 1), (1,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    eps = 1e-5  # default epsilon

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        weight = torch.randn(shape[1], device=device, dtype=dtype)
        try:
            torch_out = torch_rmsnorm_fused_llama(x, weight)
            triton_out = triton_rmsnorm_fused_llama(x, weight)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: x{shape[0]}, weight{shape[1]}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: x{shape[0]}, weight{shape[1]}: {e}")


if __name__ == "__main__":
    test_rmsnorm_fused_llama_func()
