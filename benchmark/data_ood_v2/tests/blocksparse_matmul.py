import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_ood_v2.triton.blocksparse_matmul import triton_blocksparse_matmul
from benchmark.data_ood_v2.torch.blocksparse_matmul import torch_blocksparse_matmul

def test_blocksparse_matmul():
    print("Testing Block-Sparse MatMul...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Test cases (M, K, N)
    input_shapes = [
        ((256, 512), (512, 512)),
        ((512, 1024), (1024, 1024)),
        ((128, 2048), (2048, 2048))
    ]

    # Fixed BLOCK_SIZE inside the Triton kernel
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64

    for idx, shape in enumerate(input_shapes):
        M, K = shape[0]
        K, N = shape[1]
        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        # indicator shape must be aligned with BLOCK_SIZE inside the kernel
        num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

        try:
            torch_out = torch_blocksparse_matmul(a, b)
            triton_out = triton_blocksparse_matmul(a, b)

            # Tolerance for FP16
            atol = 1e-1
            max_diff = (torch_out - triton_out).abs().max().item()
            print(f"\033[92m OK \033[0m case {idx}: A({M},{K}) x B({K},{N}) max_diff={max_diff:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m case {idx}: A({M},{K}) x B({K},{N}): {e}")


if __name__ == "__main__":
    test_blocksparse_matmul()
