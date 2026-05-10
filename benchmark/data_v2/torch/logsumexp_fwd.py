import torch
from typing import Optional

def torch_logsumexp_fwd(x: torch.Tensor, scale: Optional[float] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Pure PyTorch implementation of logsumexp over the last dimension, with optional scaling.
    
    Args:
        x (Tensor): Input tensor of any shape.
        scale (Optional[float]): Optional scale applied to the input tensor.
        dtype (Optional[torch.dtype]): Output dtype.
    
    Returns:
        Tensor: logsumexp over the last dimension.
    """
    # Scale in float32 first (matches Triton which loads as float32 before scaling)
    x_f = x.float()
    if scale is not None:
        x_f = x_f * scale
    result = torch.logsumexp(x_f, dim=-1)
    if dtype is not None and dtype != torch.float:
        result = result.to(dtype)
    return result
