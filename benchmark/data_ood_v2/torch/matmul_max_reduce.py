import torch

def torch_matmul_max_reduce(a, b):
    """
    Torch reference implementation for max-reduce matmul:
    C[i, j] = max_k(A[i, k] * B[k, j])
    """
    # Optimized using broadcasting
    a_exp = a.unsqueeze(1)          # (M, 1, K)
    b_exp = b.T.unsqueeze(0)        # (1, N, K)
    all_products = a_exp * b_exp    # (M, N, K)
    result = torch.max(all_products, dim=2).values
    return result
