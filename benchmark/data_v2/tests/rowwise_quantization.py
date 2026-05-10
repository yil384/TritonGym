# benchmark/data/tests/rowwise_quantization_test.py
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rowwise_quantization import triton_rowwise_quantization
from benchmark.data_v2.torch.rowwise_quantization import torch_rowwise_quantization

def test_rowwise_quantization():
    print("Testing rowwise_quantization...")

    input_shapes = [
        ((2, 3),),
        ((4, 8),),
        ((2, 5),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_rowwise_quantization(x)
            triton_out = triton_rowwise_quantization(x)
            max_abs = (torch_out.float() - triton_out.float()).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: {e}")

if __name__ == "__main__":
    test_rowwise_quantization()
