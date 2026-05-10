import torch
from typing import Optional

def torch_reversed_cumsum(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Pure PyTorch implementation of Triton 4D reversed cumulative sum along last two dims.
    
    Args:
        s: input tensor of shape (B, H, T, S)
        dtype: optional output dtype
    Returns:
        Tensor of same shape as input
    """
    dtype = dtype or s.dtype
    B, H, T, S = s.shape

    # Reversed cumulative sum along the T dimension (dim=-2)
    # For each position t, out[..., t, :] = sum_{k=t}^{T-1} s[..., k, :]
    out = torch.flip(torch.cumsum(torch.flip(s, dims=[-2]), dim=-2), dims=[-2])
    return out.to(dtype)
