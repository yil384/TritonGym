import torch

def torch_softmax_optimize(x: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch version of Triton-optimized softmax along the last dimension.
    """
    return torch.softmax(x, dim=-1)
