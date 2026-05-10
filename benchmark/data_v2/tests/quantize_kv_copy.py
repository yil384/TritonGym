import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.quantize_kv_copy import triton_quantize_kv_copy
from benchmark.data_v2.torch.quantize_kv_copy import torch_quantize_kv_copy

def test_quantize_kv_copy():
    print("Testing quantize_kv_copy operator...")

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
        seq_len, head_num, head_dim = shape[0]
        quant_group_dim = 8
        group_size = head_dim // quant_group_dim

        DestLoc = torch.arange(seq_len, device=device, dtype=torch.int32)

        Out_triton = torch.empty((seq_len, head_num, head_dim), device=device, dtype=torch.int8)
        Out_scale_triton = torch.empty((seq_len, head_num, group_size), device=device, dtype=torch.float32)

        Out_torch = torch.empty((seq_len, head_num, head_dim), device=device, dtype=torch.int8)
        Out_scale_torch = torch.empty((seq_len, head_num, group_size), device=device, dtype=torch.float32)

        try:
            torch_quantize_kv_copy(K, DestLoc, Out_torch, Out_scale_torch)
            triton_quantize_kv_copy(K, DestLoc, Out_triton, Out_scale_triton)

            max_abs_scale = (Out_scale_torch - Out_scale_triton).abs().max().item()
            max_abs_out = (Out_torch.float() - Out_triton.float()).abs().max().item()

            print(f"\033[92m OK \033[0m quantize_kv_copy case {idx}: input{shape} "
                  f"max_diff_scale={max_abs_scale:.3e} max_diff_out={max_abs_out:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m quantize_kv_copy case {idx}: input{shape}: {e}")

if __name__ == "__main__":
    test_quantize_kv_copy()
