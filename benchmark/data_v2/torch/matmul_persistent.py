import torch

def torch_matmul_persistent(a: torch.Tensor, b: torch.Tensor):
    """
    PyTorch implementation of Triton persistent matmul kernel.
    The Triton kernel accumulates in fp32 then casts to fp16 (for fp16 input).

    Args:
        a: [M, K]
        b: [K, N]
    Returns:
        c: [M, N] in same dtype as input
    """
    dtype = a.dtype
    # Use native matmul which matches Triton's tensor core accumulation pattern
    c = torch.matmul(a, b)
    return c.to(dtype)
