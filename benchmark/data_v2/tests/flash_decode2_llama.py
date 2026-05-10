import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.flash_decode2_llama import triton_flash_decode2_llama
from benchmark.data_v2.torch.flash_decode2_llama import torch_flash_decode2_llama

def test_flash_decode_stage2():
    print("Testing flash_decode_stage2 operator...")

    input_shapes = [
        ((2, 4, 3, 32), (2, 4, 3), (2, 4, 32)),
        ((3, 1, 2, 64), (3, 1, 2), (3, 1, 64)),
        ((1, 8, 5, 16), (1, 8, 5), (1, 8, 16)),
        ((4, 2, 4, 128), (4, 2, 4), (4, 2, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        mid_out = torch.randn(shape[0], device=device, dtype=dtype)
        mid_out_logexpsum = torch.randn(shape[1], device=device, dtype=dtype)
        O = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            O_torch = torch_flash_decode2_llama(mid_out, mid_out_logexpsum, O)
            O_triton = triton_flash_decode2_llama(mid_out, mid_out_logexpsum, O)
            max_abs = (O_torch - O_triton).abs().max().item()
            print(f"\033[92m OK \033[0m flash_decode_stage2 case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m flash_decode_stage2 case {idx}: {e}")

if __name__ == "__main__":
    test_flash_decode_stage2()
