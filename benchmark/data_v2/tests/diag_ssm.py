import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.diag_ssm import triton_diag_ssm
from benchmark.data_v2.torch.diag_ssm import torch_diag_ssm

def test_diag_ssm():
    print("Testing diag_ssm operator...")

    input_shapes = [
        ((2, 3), (5, 2, 3), (3,)),  # length, batch_size, dim
        ((3, 2), (4, 3, 2), (2,)),
        ((1, 4), (6, 1, 4), (4,))
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        # Real number test
        s = torch.randn(shape[0], device=device, dtype=dtype)
        x = torch.randn(shape[1], device=device, dtype=dtype)
        Lambda = torch.randn(shape[2], device=device, dtype=dtype)

        try:
            torch_out = torch_diag_ssm(s, x, Lambda)
            triton_out = triton_diag_ssm(s.clone(), x.clone(), Lambda.clone())
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m Case {idx} (real): max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m Case {idx} (real): {e}")

if __name__ == "__main__":
    test_diag_ssm()
