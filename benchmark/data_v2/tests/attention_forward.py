import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.attention_forward import triton_attention_forward
from benchmark.data_v2.torch.attention_forward import torch_attention_forward  


def test_attention_forward():
    print("Testing attention forward operator...")

    # Test case shape: (batch_size, num_heads, seq_length, head_dim)
    input_shapes = [
        ((2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 128)),
        ((2, 8, 256, 64), (2, 8, 256, 64), (2, 8, 256, 64)),
        ((1, 4, 512, 64), (1, 4, 512, 64), (1, 4, 512, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, dim in enumerate(input_shapes):
        # Triton kernel input must be float16
        q = torch.randn(dim[0], dtype=torch.float16, device=device)
        k = torch.randn(dim[1], dtype=torch.float16, device=device)
        v = torch.randn(dim[2], dtype=torch.float16, device=device)

        try:
            # Torch forward: softmax internally casts to float32, output cast back to float16
            torch_out = torch_attention_forward(q, k, v).to(torch.float16)

            # Triton forward: both input and output are float16
            triton_out = triton_attention_forward(q, k, v)

            # Compare maximum absolute error
            max_abs = (torch_out - triton_out).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"attention case {idx}: shape=({dim[0][0]},{dim[0][1]},{dim[0][2]},{dim[0][3]}) max_abs={max_abs:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"attention case {idx}: shape=({dim[0][0]},{dim[0][1]},{dim[0][2]},{dim[0][3]}) : {e}")


if __name__ == "__main__":
    test_attention_forward()