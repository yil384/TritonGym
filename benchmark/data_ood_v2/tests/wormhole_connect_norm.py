import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.triton.wormhole_connect_norm import triton_wormhole_connect_norm
from benchmark.data_ood_v2.torch.wormhole_connect_norm import torch_wormhole_connect_norm

def test_wormhole_connect_norm():
    print("Testing Quantum Wormhole Connect Norm operator...")
    input_shapes = [
        ((32, 256), (256,), (256,)),
        ((16, 512), (512,), (512,)),
        ((8, 1024), (1024,), (1024,)),
        ((4, 2048), (2048,), (2048,)),
    ]
    dtype_list = [torch.float32, torch.float16]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, shape in enumerate(input_shapes):
        rows, cols = shape[0]
        for dtype in dtype_list:
            x = torch.randn(rows, cols, device=device, dtype=dtype)
            w = torch.randn(cols, device=device, dtype=dtype)
            b = torch.randn(cols, device=device, dtype=dtype)
            try:
                torch_out = torch_wormhole_connect_norm(x, w, b)
                triton_out = triton_wormhole_connect_norm(x, w, b)
                max_abs = (torch_out - triton_out).abs().max().item()
                print(f"\033[92m OK \033[0m", end=' ')
                print(f"case {idx}: shape={shape}, dtype={dtype} max_abs={max_abs:.3e}")
            except Exception as e:
                print(f"\033[91m ERROR \033[0m", end=' ')
                print(f"case {idx}: shape={shape}, dtype={dtype}: {e}")

if __name__ == "__main__":
    test_wormhole_connect_norm()
