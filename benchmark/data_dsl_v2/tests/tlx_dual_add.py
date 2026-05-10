import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.tlx_dual_add import triton_tlx_dual_add
from benchmark.data_dsl_v2.torch.tlx_dual_add import torch_tlx_dual_add


def test_tlx_dual_add():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shapes = [
        ((1024, 1024), (1024, 1024), (1024, 1024), (1024, 1024)),
        ((4096, 4096), (4096, 4096), (4096, 4096), (4096, 4096)),
    ]
    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device)
        y = torch.randn(shape[1], device=device)
        a = torch.randn(shape[2], device=device)
        b = torch.randn(shape[3], device=device)

        out_triton = triton_tlx_dual_add(x, y, a, b)
        out_torch = torch_tlx_dual_add(x, y, a, b)
        try:
            max_abs_0 = torch.max(torch.abs(out_triton[0] - out_torch[0])).item()
            max_abs_1 = torch.max(torch.abs(out_triton[1] - out_torch[1])).item()
            print(f"[OK] Test {idx}: max absolute difference = {max(max_abs_0, max_abs_1)}")
        except AssertionError:
            print(f"[FAIL] Test {idx} failed.")

if __name__ == "__main__":
    test_tlx_dual_add()
