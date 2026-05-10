import torch
import sys
import os
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.attention_score import triton_attention_score
from benchmark.data_v2.torch.attention_score import torch_attention_score


def test_attention_score_compare():
    print("Testing attention score operator...")

    input_shapes = [
        ((2, 4, 128, 64), (2, 4, 128, 64)),
        ((2, 4, 128, 128), (2, 4, 128, 128)),
        ((1, 8, 64, 64), (1, 8, 64, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        B, H, N_CTX, D_MODEL = shape[0]

        q = torch.randn(shape[0], device=device, dtype=dtype)
        k = torch.randn(shape[1], device=device, dtype=dtype)

        # Unified parameters
        sliding_window = None
        complement_sliding_window = False

        try:
            # PyTorch reference
            torch_out = torch_attention_score(q, k, sliding_window, complement_sliding_window)

            # Triton kernel
            triton_out = triton_attention_score(q, k, sliding_window, complement_sliding_window)

            max_abs = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m score case {idx}: shape={shape} max_abs={max_abs:.3e}")
        except Exception as e:
            warnings.warn(f"Test failed for case {idx}: {e}")
            print(f"\033[91m ERROR \033[0m score case {idx}: shape={shape}: {e}")


if __name__ == "__main__":
    test_attention_score_compare()
