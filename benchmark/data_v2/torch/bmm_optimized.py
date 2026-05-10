import torch

def torch_bmm_optimized(A, B):
    """Perform batched matrix multiplication over contiguous inputs.

    Args:
        A: Input tensor of shape (batch, M, K).
        B: Input tensor of shape (batch, K, N).

    Returns:
        Output tensor of shape (batch, M, N).
    """
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    for i in range(batch):
        out[i] = torch.matmul(A[i], B[i])
    return out

