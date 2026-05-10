import torch

def torch_square_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch version of triton_square_matrix: element-wise square.
    """
    return x * x
