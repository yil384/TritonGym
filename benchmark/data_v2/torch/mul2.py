import torch

def torch_mul2(x: torch.Tensor, BLOCK_SIZE: int = 16) -> torch.Tensor:
    """
    Multiply each element by 2 (returns new tensor).
    BLOCK_SIZE is for triton signature compatibility only.
    """
    return x * 2

def torch_mul2_inplace(x: torch.Tensor, BLOCK_SIZE: int = 16) -> torch.Tensor:
    """
    Multiply each element by 2 (in-place).
    BLOCK_SIZE is for triton signature compatibility only.
    """
    x.mul_(2)
    return x
