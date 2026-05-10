import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.torch.quantum_fluctuation_norm import torch_quantum_fluctuation_norm
from benchmark.data_ood_v2.triton.quantum_fluctuation_norm import triton_quantum_fluctuation_norm

def test_quantum_fluctuation_norm():
    print("Testing Quantum Fluctuation Norm operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Shapes for testing: list of (n_rows, n_cols)
    input_shapes = [
        ((32, 256), (32, 256), (32, 256)),
        ((16, 512), (16, 512), (16, 512)),
        ((8, 1024), (8, 1024), (8, 1024)),
        ((4, 2048), (4, 2048), (4, 2048)),
    ]

    for idx, shape in enumerate(input_shapes):
        X = torch.randn(shape[0], device=device, dtype=dtype)
        W = torch.randn(shape[1], device=device, dtype=dtype)
        B = torch.randn(shape[2], device=device, dtype=dtype)
        try:
            torch_Y = torch_quantum_fluctuation_norm(X, W, B)
            triton_Y = triton_quantum_fluctuation_norm(X, W, B)
            max_abs = (torch_Y - triton_Y).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: shape=({shape[0]}), dtype={dtype} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: shape=({shape[0]}), dtype={dtype}: {e}")

if __name__ == "__main__":
    test_quantum_fluctuation_norm()
