import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.torch.matmul_max_reduce import torch_matmul_max_reduce
from benchmark.data_ood_v2.triton.matmul_max_reduce import triton_matmul_max_reduce

def test_matmul_max_reduce():
    print("Testing MatMul Max-Reduce operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Shapes: (M, K, N)
    input_shapes = [
        ((128, 256), (256, 128)),
        ((256, 512), (512, 256)),
        ((512, 256), (256, 512)),
    ]

    for idx, shape in enumerate(input_shapes):
        a = torch.randn(shape[0], device=device, dtype=dtype)
        b = torch.randn(shape[1], device=device, dtype=dtype)
        try:
            torch_out = torch_matmul_max_reduce(a, b)
            triton_out = triton_matmul_max_reduce(a, b)
            max_diff = (torch_out - triton_out).abs().max().item()
            
            print(f"\033[92m OK \033[0m case {idx}: M={shape[0][0]}, K={shape[0][1]}, N={shape[1][1]}, max_diff={max_diff:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: M={shape[0][0]}, K={shape[0][1]}, N={shape[1][1]}: {e}")

if __name__ == "__main__":
    test_matmul_max_reduce()
