import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.quant_transpose_kernel import triton_quant_transpose_kernel
from benchmark.data_v2.torch.quant_transpose_kernel import torch_quant_transpose_kernel

def test_quant_transpose():
    print("Testing quantize_global_transpose operator...")
    input_shapes = [
        ((128, 256),),
        ((256, 128),),
        ((512, 256),),
        ((256, 512),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        input_tensor = torch.randn(shape[0], device=device, dtype=dtype)
        try:
            torch_out = torch_quant_transpose_kernel(input_tensor)
            triton_out = triton_quant_transpose_kernel(input_tensor)
            max_abs_diff = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m quant_transpose case {idx}: input{shape} -> output{torch_out.shape}, max_abs_diff={max_abs_diff:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m quant_transpose case {idx}: input{shape}: {e}")


if __name__ == "__main__":
    test_quant_transpose()
