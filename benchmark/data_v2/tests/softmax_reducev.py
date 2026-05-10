import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.softmax_reducev import triton_softmax_reducev
from benchmark.data_v2.torch.softmax_reducev import torch_softmax_reducev

def test_softmax_reducev():
    print("Testing softmax_reducev operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Safe input shapes
    input_shapes = [
        ((2, 256), (2, 128, 64)),  # batch, head, max_seq_len, d_model
        ((4, 192), (4, 64, 32)),
    ]

    for idx, batch in enumerate(input_shapes):
        Logics = torch.randn(batch[0], device=device, dtype=dtype)
        V = torch.randn(batch[1], device=device, dtype=dtype)

        # Triton
        O_triton = triton_softmax_reducev(Logics, V)
        
        # Torch
        O_torch = torch_softmax_reducev(Logics, V)

        # Compare
        max_abs = (O_triton - O_torch).abs().max().item()
        print(f"\033[92m OK \033[0m softmax_reducev case {idx}: max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_softmax_reducev()
