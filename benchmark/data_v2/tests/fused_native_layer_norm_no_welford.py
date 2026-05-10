import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fused_native_layer_norm_no_welford import triton_fused_native_layer_norm_no_welford
from benchmark.data_v2.torch.fused_native_layer_norm_no_welford import torch_fused_native_layer_norm_no_welford

def test_fused_native_layer_norm_no_welford():
    print("Testing fused_native_layer_norm_no_welford operator...")

    # Shapes for input
    input_shapes = [
        ((4096,), (4096,), (128, 4096)),
        ((2048,), (2048,), (256, 2048)),
        ((8192,), (8192,), (64, 8192)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        weight = torch.randn(shape[0], dtype=dtype, device=device)
        bias = torch.randn(shape[1], dtype=dtype, device=device)
        x = torch.randn(shape[2], dtype=dtype, device=device)
        try:
            torch_out = torch_fused_native_layer_norm_no_welford(weight, bias, x)
            triton_out = triton_fused_native_layer_norm_no_welford(weight, bias, x)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_fused_native_layer_norm_no_welford()
