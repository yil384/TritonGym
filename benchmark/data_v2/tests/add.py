import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.add import triton_add
from benchmark.data_v2.torch.add import torch_add

def test_add():
    print("Testing add operator...")
    # Shapes for (input, other)
    input_shapes = [
        ((1024, 1024), (1024, 1024)),
        ((4096, 4096), (4096, 4096)),  # may fail: not broadcastable
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    def parse_pair(entry):
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and all(isinstance(x, (list, tuple)) for x in entry):
            return tuple(entry[0]), tuple(entry[1])
        raise ValueError(f"Invalid input shape entry: {entry}")

    for idx, entry in enumerate(input_shapes):
        Ashape, Bshape = parse_pair(entry)
        a = torch.randn(*Ashape, device=device, dtype=dtype)
        b = torch.randn(*Bshape, device=device, dtype=dtype)
        try:
            torch_out = torch_add(a, b)
            triton_out = triton_add(a, b)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"add case {idx}: A{Ashape} + B{Bshape} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"add case {idx}: A{Ashape} + B{Bshape}: {e}")


if __name__ == "__main__":
    test_add()