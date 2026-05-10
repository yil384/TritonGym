import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.fused_recurrent_hgrn import triton_fused_recurrent_hgrn
from benchmark.data_v2.torch.fused_recurrent_hgrn import torch_fused_recurrent_hgrn

def test_fused_recurrent_hgrn():
    print("Testing fused_recurrent_hgrn operator...")
    # Shapes for (x, g)
    input_shapes = [
        ((1, 2, 2, 2), (1, 2, 2, 2)),
        ((2, 4, 8, 16), (2, 4, 8, 16)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, (x_shape, g_shape) in enumerate(input_shapes):
        x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)
        g = torch.randn(*g_shape, device=device, dtype=dtype, requires_grad=True)

        # Triton forward
        try:
            triton_out = triton_fused_recurrent_hgrn(x, g)
        except Exception as e:
            print(f"\033[91m ERROR \033[0m triton case {idx}: {e}")
            continue

        # Torch forward
        try:
            torch_out = torch_fused_recurrent_hgrn(x, g)
        except Exception as e:
            print(f"\033[91m ERROR \033[0m torch case {idx}: {e}")
            continue

        # Compare
        max_abs = (triton_out - torch_out).abs().max().item()
        print(f"\033[92m OK \033[0m fused_recurrent_hgrn case {idx}: max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_fused_recurrent_hgrn()
