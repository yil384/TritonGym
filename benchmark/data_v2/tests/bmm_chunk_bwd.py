import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.bmm_chunk_bwd import triton_bmm_chunk_bwd
from benchmark.data_v2.torch.bmm_chunk_bwd import torch_bmm_chunk_bwd  


def test_bmm_chunk_bwd():
    print("Testing bmm_chunk_bwd operator...")

    input_shapes = [
        ((2, 128, 1, 64), (2, 4, 1, 32, 64)),
        ((2, 128, 4, 64), (2, 4, 4, 32, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        try:
            # Build inputs
            # if ngroups == 1:
            #     a = torch.randn(batch, seqlen, k, device=device, dtype=dtype)
            #     dout = torch.randn(batch, nchunks, chunk_size, k, device=device, dtype=dtype)
            a = torch.randn(shape[0], device=device, dtype=dtype)
            dout = torch.randn(shape[1], device=device, dtype=dtype)

            torch_out = torch_bmm_chunk_bwd(a, dout)
            triton_out = triton_bmm_chunk_bwd(a, dout)

            max_abs = (torch_out - triton_out).abs().max().item()
  
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"bmm_chunk_bwd case {idx}: shape_a={shape[0]}, shape_dout={shape[1]}, max_abs={max_abs:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"bmm_chunk_bwd case {idx}: shape_a={shape[0]}, shape_dout={shape[1]}: {e}")

            
if __name__ == "__main__":
    test_bmm_chunk_bwd()
