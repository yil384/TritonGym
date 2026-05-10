import torch
from typing import Optional

def torch_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of Triton reversed cumulative sum over last dimension
    with scalar accumulation across blocks.
    
    Args:
        s: input tensor of shape (B, H, T)
        dtype: optional output dtype
    Returns:
        Tensor of same shape as input
    """
    dtype = dtype or s.dtype
    B, H, T = s.shape
    # Compute reversed cumulative sum along last dimension
    # Equivalent to: o[:, :, t] = sum_{k=t}^{T-1} s[:, :, k] - cumsum(s[:, :, t:])
    # We can do it with flip + cumsum + flip
    out = torch.flip(torch.cumsum(torch.flip(s, dims=[-1]), dim=-1), dims=[-1])
    return out.to(dtype)
