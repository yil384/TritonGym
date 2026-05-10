import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.argmax import triton_argmax
from benchmark.data_v2.torch.argmax import torch_argmax

def test_argmax():
    print("Testing argmax operator...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # input_shapes: list of tuples (shape, dim)
    input_shapes = [
        ((1024,),),
        ((1024, 1024),),
        ((64, 128, 256),),
        ((8, 16, 32, 64),),
    ]

    for idx, (shape,) in enumerate(input_shapes):
        a = torch.randn(shape, device=device, dtype=dtype)

        try:
            torch_out = torch_argmax(a)
            # Compare with triton_argmax
            triton_out = triton_argmax(a)
            max_abs = (torch_out.float() - triton_out.float()).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"argmax case {idx}: shape={shape}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"argmax case {idx}: shape={shape}: {e}")

if __name__ == "__main__":
    test_argmax()
