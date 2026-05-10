import torch

def torch_matrix_transpose(matrix: torch.Tensor) -> torch.Tensor:
    """
    Core Torch implementation of matrix transpose.
    Equivalent to the Triton kernel behavior.
    """
    return matrix.T.contiguous()
