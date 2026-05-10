import torch

def torch_matrix_vector_multip(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Core Torch implementation of matrix-vector multiplication.
    Equivalent to the Triton mv_kernel behavior.
    """
    # A: (N, M), B: (M,)
    assert A.shape[1] == B.shape[0], "incompatible dimensions"
    return torch.matmul(A, B)
