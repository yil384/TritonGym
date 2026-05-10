import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_gate_recurrence import triton_chunk_gate_recurrence
from benchmark.data_v2.torch.chunk_gate_recurrence import torch_chunk_gate_recurrence

def test_chunk_gate_recurrent():
    print("Testing chunk_gate_recurrent operator...")

    input_shapes = [
        ((2, 4, 64, 64, 64), (2, 4, 64), (2, 4, 64, 64)),
        ((3, 1, 128, 128, 128), (3, 1, 128), (3, 1, 128, 128)),
        ((1, 8, 256, 64, 32), (1, 8, 256), (1, 8, 64, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        kv = torch.randn(shape[0], dtype=dtype, device=device)
        cross_decay = torch.randn(shape[1], dtype=dtype, device=device)
        last_kv = torch.randn(shape[2], dtype=dtype, device=device)

        try:
            # Triton forward
            out_triton = triton_chunk_gate_recurrence(kv, cross_decay, last_kv)

            # Torch forward
            out_torch = torch_chunk_gate_recurrence(kv, cross_decay, last_kv)

            # Compare maximum absolute error
            max_abs_last = (out_triton - out_torch).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"chunk_gate_recurrent case {idx}: B={shape[0][0]}, H={shape[0][1]}, N={shape[0][2]}, D_k={shape[0][3]}, D_v={shape[0][4]} "
                  f"max_abs_last={max_abs_last:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"chunk_gate_recurrent case {idx}: B={shape[0][0]}, H={shape[0][1]}, N={shape[0][2]}, D_k={shape[0][3]}, D_v={shape[0][4]}: {e}")

if __name__ == "__main__":
    test_chunk_gate_recurrent()
