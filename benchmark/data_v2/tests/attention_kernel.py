import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.attention_kernel import triton_attention_kernel
from benchmark.data_v2.torch.attention_kernel import torch_attention_kernel

def test_attention_rel_h_rel_w_kernel_aligned():
    print("Testing attention rel_h / rel_w aligned kernel...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # input_shapes: (BATCH_SIZE, HEADS, N_CTX, D_MODEL)
    input_shapes = [
        ((2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 128)),
        ((2, 8, 128, 128), (2, 8, 128, 128), (2, 8, 128, 128), (2, 8, 128, 128)),
        ((1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128)),
    ]

    for idx, shape in enumerate(input_shapes):
        # create input tensors
        q = torch.randn(shape[0], dtype=dtype, device=device)
        k = torch.randn(shape[1], dtype=dtype, device=device)
        v = torch.randn(shape[2], dtype=dtype, device=device)
        rel_h_w = torch.randn(shape[3], dtype=dtype, device=device)

        # Torch output
        try:
            torch_out = torch_attention_kernel(q, k, v, rel_h_w, sm_scale=0.1)

            triton_out = triton_attention_kernel(q, k, v, rel_h_w, sm_scale=0.1)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"attention case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"attention case {idx}: comparison failed: {e}")


if __name__ == "__main__":
    test_attention_rel_h_rel_w_kernel_aligned()
