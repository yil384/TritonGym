import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.token_softmax_bloom import triton_token_softmax_bloom
from benchmark.data_v2.torch.token_softmax_bloom import torch_token_softmax_bloom

def test_token_softmax():
    print("Testing token-wise softmax operator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Shapes for (Logics, B_Start_Loc, B_Seqlen)
    input_shapes = [
        ((12, 4100),)
    ]

    for idx, shape in enumerate(input_shapes):
        Logics = torch.empty(shape[0], device=device, dtype=dtype).normal_(mean=0.1, std=10)

        try:
            torch_out = torch_token_softmax_bloom(Logics)
            triton_out = triton_token_softmax_bloom(Logics)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m", end=' ')
            print(f"softmax case {idx}: Logics{shape[0]} max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"softmax case {idx}: Logics{shape[0]}: {e}")

if __name__ == "__main__":
    test_token_softmax()
