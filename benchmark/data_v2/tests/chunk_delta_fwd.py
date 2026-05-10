import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_delta_fwd import triton_chunk_delta_fwd
from benchmark.data_v2.torch.chunk_delta_fwd import torch_chunk_delta_fwd

def test_chunk_fwd_h():
    print("Testing chunk_delta_fwd_h operator...")

    # All shapes: k=[B,H,T,K], w=[B,H,T,K], u=[B,H,T,V]
    # K must be power-of-2 and <= 256, and NK must be 1
    input_shapes = [
        ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
        ((1, 2, 64, 32), (1, 2, 64, 32), (1, 2, 64, 32)),
        ((2, 3, 32, 16), (2, 3, 32, 16), (2, 3, 32, 16)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        k = torch.randn(shape[0], dtype=dtype, device=device)
        w = torch.randn(shape[1], dtype=dtype, device=device)
        u = torch.randn(shape[2], dtype=dtype, device=device)

        try:
            # Triton output
            v_triton = triton_chunk_delta_fwd(k, w, u)
            # Torch output
            v_torch = torch_chunk_delta_fwd(k, w, u)

            max_abs_h = (v_torch - v_triton).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"chunk_delta_fwd_h case {idx}: B={shape[0][0]}, H={shape[0][1]}, T={shape[0][2]}, K={shape[0][3]}, V={shape[2][3]} "
                  f"max_abs_h={max_abs_h:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"chunk_delta_fwd_h case {idx}: B={shape[0][0]}, H={shape[0][1]}, T={shape[0][2]}, K={shape[0][3]}, V={shape[2][3]}: {e}")


if __name__ == "__main__":
    test_chunk_fwd_h()
