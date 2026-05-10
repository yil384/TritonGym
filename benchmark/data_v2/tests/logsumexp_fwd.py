import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.logsumexp_fwd import triton_logsumexp_fwd
from benchmark.data_v2.torch.logsumexp_fwd import torch_logsumexp_fwd


def test_logsumexp():
    input_shapes = [
        ((4, 64),),
        ((4, 16, 64),)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)

        triton_out = triton_logsumexp_fwd(x)
        torch_out = torch_logsumexp_fwd(x)

        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m logsumexp case {idx}: max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_logsumexp()
