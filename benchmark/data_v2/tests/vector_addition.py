import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.vector_addition import triton_vector_addition
from benchmark.data_v2.torch.vector_addition import torch_vector_addition


def test_add():
    print("Testing add operator...")
    # Shapes for vector addition
    input_shapes = [
        ((98432,), (98432,)),
        ((1024,),(1024,)),
        ((2048,), (2048,)),
        ((4096,), (4096,)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape_pair in enumerate(input_shapes):
        Ashape, Bshape = shape_pair[0], shape_pair[1]
        a = torch.rand(Ashape, device=device, dtype=dtype)
        b = torch.rand(Bshape, device=device, dtype=dtype)

        try:
            torch_out = torch_vector_addition(a, b)
            triton_out = triton_vector_addition(a, b)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"add case {idx}: A{Ashape} + B{Bshape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"add case {idx}: A{Ashape} + B{Bshape}: {e}")


if __name__ == "__main__":
    test_add()
