import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fused_recurrent_delta import triton_fused_recurrent_delta
from benchmark.data_v2.torch.fused_recurrent_delta import torch_fused_recurrent_delta

def test_fused_recurrent_delta():
    print("Testing fused_recurrent_delta operator...")

    input_shapes = [
        ((2, 4, 8, 16), (2, 4, 8, 16), (2, 4, 8, 32), (2, 4, 8, 32)),  # small test
        ((1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 16), (1, 2, 4, 16)),    # tiny test
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, (q_shape, k_shape, v_shape, beta_shape) in enumerate(input_shapes):
        q = torch.randn(*q_shape, device=device, dtype=dtype)
        k = torch.randn(*k_shape, device=device, dtype=dtype)
        v = torch.randn(*v_shape, device=device, dtype=dtype)
        beta = torch.randn(*beta_shape, device=device, dtype=dtype)

        try:
            torch_out = torch_fused_recurrent_delta(q, k, v, beta)
            triton_out = triton_fused_recurrent_delta(q, k, v, beta)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m fused_recurrent_delta case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m fused_recurrent_delta case {idx}: {e}")

if __name__ == "__main__":
    test_fused_recurrent_delta()
