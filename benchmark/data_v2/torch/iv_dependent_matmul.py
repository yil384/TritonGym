import torch

def torch_iv_dependent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    BLOCK_SIZE_M: int = 32,
    BLOCK_SIZE_N: int = 32,
    BLOCK_SIZE_K: int = 32,
    type: str = "pre_load",
    device: torch.device = "cuda",
) -> torch.Tensor:
    """
    Pure PyTorch implementation of IV Dependent MatMul.
    Matches triton signature: triton_iv_dependent_matmul(a, b, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, type, device)
    Note: The triton wrapper re-randomizes a and b internally, so this does the same to match.
    a: [M, K]
    b: [K, N]
    Returns c: [M, N] float16
    """
    M, K = a.shape
    N = b.shape[1]
    device = a.device

    # The triton wrapper re-randomizes inputs, so we do the same
    a = torch.rand((M, K), device=device)
    b = torch.rand((K, N), device=device)

    c = (a @ b).to(torch.float16)
    return c
