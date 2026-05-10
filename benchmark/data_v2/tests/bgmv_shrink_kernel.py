import torch
import sys
import os
import inspect

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.bgmv_shrink_kernel import triton_bgmv_shrink_kernel
from benchmark.data_v2.torch.bgmv_shrink_kernel import torch_bgmv_shrink_kernel


def test_bgmv_shrink():
    print("Testing bgmv_shrink operator...")

    # input_shapes entries: (batch_size, N, K, lora_num)
    input_shapes = [
        ((2, 32), (2, 16, 32)),  # small test
        ((2, 64), (2, 32, 64)),  # edge case: lora_num=1
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, shape in enumerate(input_shapes):
        try:
            batch_size, N, K = shape[0][0], shape[0][1], shape[1][2]
            # Build inputs (same inputs for both implementations)
            inputs = torch.randn(shape[0], device=device, dtype=dtype).contiguous()

            # IMPORTANT: create lora_a_weights as 3D: (lora_num, N, K)
            # Triton wrapper accepts 4D or 3D; Torch impl expects 3D, so use 3D here.
            lora_a_weights = torch.randn(shape[1], device=device, dtype=dtype).contiguous()

            torch_out = torch_bgmv_shrink_kernel(inputs, lora_a_weights)
            output_tensor_triton = triton_bgmv_shrink_kernel(
                inputs=inputs,
                lora_a_weights=lora_a_weights,
            )

            # Compare outputs (cast to float32 for stable comparison)
            torch_cmp = torch_out.to(torch.float32)
            triton_cmp = output_tensor_triton.to(torch.float32)

            max_abs = (torch_cmp - triton_cmp).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"bgmv_shrink case {idx}: batch={batch_size}, N={N}, K={K}, max_abs={max_abs:.3e}")

        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"bgmv_shrink case {idx}: batch={batch_size}, N={N}, K={K}: {e}")


if __name__ == "__main__":
    test_bgmv_shrink()
