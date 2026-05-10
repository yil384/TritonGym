import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.torch.l4_norm import torch_l4_norm
from benchmark.data_ood_v2.triton.l4_norm import triton_l4_norm

def test_l4_norm():
    print("Testing L4-Norm operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else torch.float16

    # Shapes for testing: list of (batch_size, feature_size)
    input_shapes = [
        ((4, 256), (256,)),
        ((8, 512), (512,)),
        ((2, 1024), (1024,)),
        ((4, 2048), (2048,)),
        ((1, 768), (768,)),
    ]

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        weight = torch.randn(shape[1], device=device, dtype=dtype)
        try:
            torch_out = torch_l4_norm(x, weight)
            triton_out = triton_l4_norm(x, weight)
            # Set tolerance
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape=({shape[0][0]},{shape[0][1]}), dtype={dtype} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape=({shape[0][0]},{shape[0][1]}), dtype={dtype}: {e}")

if __name__ == "__main__":
    test_l4_norm()
