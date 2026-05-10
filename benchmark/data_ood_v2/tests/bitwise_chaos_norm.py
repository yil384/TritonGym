import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.triton.bitwise_chaos_norm import triton_algorithmic_chaos_norm
from benchmark.data_ood_v2.torch.bitwise_chaos_norm import torch_algorithmic_chaos_norm

def test_algorithmic_chaos_norm():
    print("Testing Algorithmic Chaos Norm...")
    
    input_shapes = [
        ((128, 256), (256,), (256,), (256,)),
        ((64, 1024), (1024,), (1024,), (1024,)),
        ((32, 2048), (2048,), (2048,), (2048,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        N_ROWS, N_COLS = shape[0]
        x = torch.rand(shape[0], device=device, dtype=dtype) * 10
        w1 = torch.randn(shape[1], device=device, dtype=dtype)
        w2 = torch.randn(shape[2], device=device, dtype=dtype)
        b = torch.randn(shape[3], device=device, dtype=dtype)

        x = x.contiguous()

        try:
            torch_out = torch_algorithmic_chaos_norm(x, w1, w2, b)
            triton_out = triton_algorithmic_chaos_norm(x, w1, w2, b)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: shape={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: shape={shape}: {e}")


if __name__ == "__main__":
    test_algorithmic_chaos_norm()
