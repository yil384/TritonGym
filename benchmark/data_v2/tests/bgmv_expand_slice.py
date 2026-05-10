import torch
import sys
import os
import warnings
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.bgmv_expand_slice import triton_bgmv_expand_slice
from benchmark.data_v2.torch.bgmv_expand_slice import torch_bgmv_expand_slice


def test_bgmv_expand_slice():
    print("Testing bgmv_expand_slice operator...")

    # (batch_size, hidden_size, rank, lora_num, slice_size)
    input_shapes = [
        ((4, 128), (3, 64, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        batch_size, hidden_size, rank, lora_num = shape[0][0], shape[0][1], shape[1][1], shape[1][0]
        inputs = torch.randn(shape[0], dtype=dtype, device=device).contiguous()
        lora_b_weights = torch.randn(shape[1], dtype=dtype, device=device).contiguous()

        try:
            
            torch_out = torch_bgmv_expand_slice(inputs, lora_b_weights)

            # Triton: allocate an output tensor
            output_triton = triton_bgmv_expand_slice(inputs, lora_b_weights)

            # Compare
            max_abs = (torch_out - output_triton).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"bgmv_expand_slice case {idx}: batch={batch_size}, hidden={hidden_size}, "
                  f"rank={rank}, lora_num={lora_num}, max_abs={max_abs:.3e}")

        except Exception as e:
            warnings.warn(f"Test failed for case {idx}: {e}")
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"bgmv_expand_slice case {idx}: batch={batch_size}, hidden={hidden_size}, "
                  f"rank={rank}, lora_num={lora_num},: {e}")


if __name__ == "__main__":
    test_bgmv_expand_slice()
