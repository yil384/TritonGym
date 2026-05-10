import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fast_ce_loss import triton_fast_ce_loss
from benchmark.data_v2.torch.fast_ce_loss import torch_fast_ce_loss

def test_fast_ce_loss():
    print("Testing fast cross-entropy operator...")

    input_shapes = [
        ((2, 3, 5),),    # batch=2, seq_len=3, vocab_size=5
        ((4, 5, 10),),   # batch=4, seq_len=5, vocab_size=10
        ((8, 6, 16),)   # batch=8, seq_len=6, vocab_size=16
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        logits = torch.randn(shape[0], device=device, dtype=dtype, requires_grad=True)

        try:
            torch_out = torch_fast_ce_loss(logits)
            triton_out = triton_fast_ce_loss(logits)
            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m fast_ce case {idx}, max_abs={max_abs:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m fast_ce case {idx}: {e}")

if __name__ == "__main__":
    test_fast_ce_loss()
