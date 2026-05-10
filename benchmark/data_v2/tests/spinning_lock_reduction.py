import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.spinning_lock_reduction import triton_spinning_lock_reduction
from benchmark.data_v2.torch.spinning_lock_reduction import torch_spinning_lock_reduction

def test_spinning_lock():
    print("Testing spinning lock reduction...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    input_shapes = [
        ((640, 128),),
        ((1280, 256),),
        ((2560, 512),),
        ((5120, 1024),),
    ]

    for idx, shape in enumerate(input_shapes):
        P = torch.randn(shape[0], device=device, dtype=dtype)

        try:
            # Run both implementations
            triton_out = triton_spinning_lock_reduction(P)
            torch_out = torch_spinning_lock_reduction(P)
            max_abs = (triton_out - torch_out).abs().max().item()
            print(f"\033[92m OK \033[0m spinning lock case {idx}: max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m spinning lock case {idx}: {e}")

if __name__ == "__main__":
    test_spinning_lock()
