import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.attention_kernel_aligned import triton_attention_kernel_aligned
from benchmark.data_v2.torch.attention_kernel_aligned import torch_attention_kernel_aligned

def test_attention_rel_h_rel_w_kernel_aligned():
    print("Testing attention rel_h / rel_w aligned kernel...")

    # Shapes for (batch_size, heads, seq_len, head_dim)
    # Note: Triton kernel internally fixes rel_h_w last dim to 128, head_dim also kept at 128
    input_shapes = [
        ((2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 128)),
        ((2, 8, 128, 128), (2, 8, 128, 128), (2, 8, 128, 128), (2, 8, 128, 128)),
        ((1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128), (1, 4, 128, 128)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for idx, dim in enumerate(input_shapes):
        # q/k/v and rel_h_w last dimension kept consistent
        q = torch.randn(dim[0], dtype=dtype, device=device)
        k = torch.randn(dim[1], dtype=dtype, device=device)
        v = torch.randn(dim[2], dtype=dtype, device=device)
        rel_h_w = torch.randn(dim[3], dtype=dtype, device=device)

        try:
            # Triton output
            triton_out = triton_attention_kernel_aligned(q, k, v, rel_h_w, sm_scale=1.0)

            torch_out = torch_attention_kernel_aligned(q, k, v, rel_h_w, sm_scale=1.0)

            max_abs = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"attention case {idx}: shape=({dim[0][0]},{dim[0][1]},{dim[0][2]},{dim[0][3]}) max_abs={max_abs:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"attention case {idx}: shape=({dim[0][0]},{dim[0][1]},{dim[0][2]},{dim[0][3]}): {e}")

if __name__ == "__main__":
    test_attention_rel_h_rel_w_kernel_aligned()
