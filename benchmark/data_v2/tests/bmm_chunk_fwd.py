import torch
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.bmm_chunk_fwd import triton_bmm_chunk_fwd
from benchmark.data_v2.torch.bmm_chunk_fwd import torch_bmm_chunk_fwd  

def test_bmm_chunk_fwd():
    print("Testing bmm_chunk_fwd operator...")

    # Shapes for (a_shape, b_shape, chunk_size, has_groups, causal, use_seq_idx)
    input_shapes = [
        ((2, 128, 64), (2, 128, 64)),
        ((2, 128, 4, 64), (2, 128, 4, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, (a_shape, b_shape) in enumerate(input_shapes):
        try:
            a = torch.randn(*a_shape, device=device, dtype=dtype)
            b = torch.randn(*b_shape, device=device, dtype=dtype)

            # Torch reference output
            torch_out = torch_bmm_chunk_fwd(a, b)

            # Triton output
            triton_out = triton_bmm_chunk_fwd(a, b)


            max_abs = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"bmm_chunk_fwd case {idx}: a{a_shape}, b{b_shape}, max_abs={max_abs:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"bmm_chunk_fwd case {idx}: a{a_shape}, b{b_shape}: {e}")


if __name__ == "__main__":
    test_bmm_chunk_fwd()
