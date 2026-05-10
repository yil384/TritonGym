import torch
import math

def torch_l4_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of L4 Normalization.
    Caution: use float32 internally for numerical stability.
    """
    # Compute sum of x^4
    sum4 = x.to(torch.float32).pow(4).sum(-1, keepdim=True)
    # Compute (sum4 + eps)^(1/4)
    norm4 = (sum4 + eps).pow(0.25)
    # Normalize and apply weights
    return x * (1 / norm4).to(x.dtype) * weight
