import torch

def torch_blocksparse_matmul(a, b):
    """Compute block-sparse matrix multiplication using dense matmul with a random block mask.

    Performs a full matmul of a and b, then zeroes out blocks where a randomly
    generated indicator matrix is 0, simulating block-sparse computation.

    Args:
        a: Left input matrix of shape (M, K).
        b: Right input matrix of shape (K, N).

    Returns:
        Block-sparse result tensor of shape (M, N).
    """
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    # Indicator shape must align with the kernel's BLOCK_SIZE
    M, K = a.shape
    K, N = b.shape
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    indicator = (torch.rand(num_blocks_m, num_blocks_n, device=a.device) > 0.5).int()
    """
    Verify block-sparse matrix multiplication correctness using dense matmul + masking.
    """
    # Step 1: Compute full dense result
    c_dense = torch.matmul(a, b)

    # Step 2: Create global mask
    mask = indicator.repeat_interleave(BLOCK_SIZE_M, dim=0).repeat_interleave(BLOCK_SIZE_N, dim=1)

    # Step 3: Apply mask
    c_masked = c_dense * mask
    return c_masked
