import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.matrix_vector_multip import triton_matrix_vector_multip
from benchmark.data_v2.torch.matrix_vector_multip import torch_matrix_vector_multip

def test_matrix_vector_multip():
    print("Testing matrix_vector_multip operator...")

    # Shapes for testing
    input_shapes = [
        ((4, 3), (3,)),
        ((32, 16), (16,)),
        ((128, 64), (64,)),
        ((256, 128), (128,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        Ashape, Bshape = shape
        A = torch.randn(*Ashape, device=device, dtype=dtype)
        B = torch.randn(*Bshape, device=device, dtype=dtype)

        try:
            torch_out = torch_matrix_vector_multip(A, B)
            triton_out = triton_matrix_vector_multip(A, B)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"matrix_vector_multip case {idx}: A{Ashape} x B{Bshape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"matrix_vector_multip case {idx}: A{Ashape} x B{Bshape}: {e}")


if __name__ == "__main__":
    test_matrix_vector_multip()
