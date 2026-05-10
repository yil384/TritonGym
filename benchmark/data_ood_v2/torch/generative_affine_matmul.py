# benchmark/data/torch/generative_affine_matmul.py
import torch

def torch_generative_affine_matmul(a, alpha=0.01, beta=0.02):
    """Multiply input by a procedurally generated affine weight matrix.

    Constructs a weight matrix B where B[k, n] = k * alpha + n * beta, then
    computes a @ B.

    Args:
        a: Input matrix of shape (M, K).
        alpha: Coefficient for the row index in the generated weight.
        beta: Coefficient for the column index in the generated weight.

    Returns:
        Output matrix of shape (M, K), since N equals K.
    """
    M, K = a.shape
    N = K
    k_indices = torch.arange(K, device=a.device, dtype=torch.float32)
    n_indices = torch.arange(N, device=a.device, dtype=torch.float32)
    b_prime = (k_indices[:, None] * alpha + n_indices[None, :] * beta).to(a.dtype)
    return torch.matmul(a, b_prime)
