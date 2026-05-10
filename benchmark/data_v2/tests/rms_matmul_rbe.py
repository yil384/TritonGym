import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.rms_matmul_rbe import triton_rms_matmul_rbe
from benchmark.data_v2.torch.rms_matmul_rbe import torch_rms_matmul_rbe

def test_rms_matmul_rbe():
    print("Testing rms_matmul_rbe operator...")

    input_shapes = [
        ((1, 16, 4096), (4096, 4096), (4096, 4096), (4096, 4096), (4096,)),
        ((8, 16, 4096), (4096, 4096), (4096, 4096), (4096, 4096), (4096,)),
        ((2, 64, 4096), (4096, 4096), (4096, 4096), (4096, 4096), (4096,)),
        ((4, 64, 4096), (4096, 4096), (4096, 4096), (4096, 4096), (4096,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        q_weight = torch.randn(shape[1], device=device, dtype=dtype)
        k_weight = torch.randn(shape[2], device=device, dtype=dtype)
        v_weight = torch.randn(shape[3], device=device, dtype=dtype)
        rms_w = torch.randn(shape[4], device=device, dtype=dtype)

        try:
            torch_q= torch_rms_matmul_rbe(
                x, q_weight, k_weight, v_weight, rms_w
            )
            triton_q= triton_rms_matmul_rbe(
                x, q_weight, k_weight, v_weight, rms_w
            )
            max_abs_q = (torch_q - triton_q).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"case {idx}: q_max_abs={max_abs_q:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"case {idx}: {e}")

if __name__ == "__main__":
    test_rms_matmul_rbe()
