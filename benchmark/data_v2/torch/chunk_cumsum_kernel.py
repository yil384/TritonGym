import torch
from typing import Optional

def torch_chunk_cumsum_kernel(s: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Compute cumulative sum along the time dimension for each batch and head.

    Args:
        s: Input tensor of shape (B, H, T).
        dtype: Optional output dtype; defaults to the dtype of s.

    Returns:
        Tensor of shape (B, H, T) containing the cumulative sum along dim T.
    """
    B, H, T = s.shape
    dtype = dtype or s.dtype
    z = torch.empty_like(s, dtype=dtype)
    for b in range(B):
        for h in range(H):
            z_bh = torch.cumsum(s[b, h], dim=0)
            z[b, h] = z_bh
    return z
