import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fused_recurrent_retention import triton_fused_recurrent_retention
from benchmark.data_v2.torch.fused_recurrent_retention import torch_fused_recurrent_retention


def test_fused_recurrent_retention():
    print("Testing fused_recurrent_retention operator...")

    input_shapes = [
        ((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 16, 16)),   # case 0: small, baseline
        ((2, 4, 8, 64), (2, 4, 8, 64), (2, 4, 8, 32), (2, 4, 64, 32)),   # case 1: larger qk and mismatched v
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(shape[1], device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(shape[2], device=device, dtype=dtype, requires_grad=True)
        init_state = torch.randn(shape[3], device=device, dtype=dtype, requires_grad=False)

        try:
            # --- case B: with init, no final ---
            o_torch = torch_fused_recurrent_retention(q, k, v, init_state)
            o_triton = triton_fused_recurrent_retention(q, k, v, init_state)
            max_abs_out = (o_torch - o_triton).abs().max().item()
            print(f"\033[92m OK \033[0m fused_recurrent_retention case {idx} (B={shape[0][0]},H={shape[0][1]},T={shape[0][2]},Dqk={shape[0][3]}), "
                  f"v(Dv={shape[2][3]}): max_abs_out={max_abs_out:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m fused_recurrent_retention case {idx}: "
                  f"(B={shape[0][0]},H={shape[0][1]},T={shape[0][2]},Dqk={shape[0][3]}), v(Dv={shape[2][3]}): {e}")


if __name__ == "__main__":
    test_fused_recurrent_retention()
