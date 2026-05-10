import torch

def torch_matrix_reduction(matrix: torch.Tensor, dtype_str: str = 'float16') -> torch.Tensor:
    """
    Torch implementation of max-reduction along axis=1.
    """
    return matrix.max(dim=1)[0]
