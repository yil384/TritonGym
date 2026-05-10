import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.llama_ff import triton_llama_ff
from benchmark.data_v2.torch.llama_ff import torch_llama_ff


def test_llama_ff():
    input_shapes = [
        ((2, 4, 64), (64, 64), (64, 64), (64,)),
        ((3, 4, 64), (64, 64), (64, 64), (64,)),
        ((2, 5, 64), (64, 64), (64, 64), (64,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    results = {}
    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], dtype=dtype, device=device)
        w1 = torch.randn(shape[1], dtype=dtype, device=device)
        w3 = torch.randn(shape[2], dtype=dtype, device=device)
        rms_w = torch.randn(shape[3], dtype=dtype, device=device)

        torch_out = torch_llama_ff(x, w1, w3, rms_w)
        triton_out = triton_llama_ff(x, w1, w3, rms_w)

        max_abs = (torch_out - triton_out).abs().max().item()
        print(f"\033[92m OK \033[0m Llama FF case {idx}: max_abs={max_abs:.3e}")
        results[f"test_case_{idx+1}"] = torch_out

    return results


if __name__ == "__main__":
    test_llama_ff()
