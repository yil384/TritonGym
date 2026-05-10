import torch
from typing import Optional

def torch_chunk_cumsum_vector(s: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Compute cumulative sum along the time dimension for each batch, head, and state.

    Args:
        s: Input tensor of shape (B, H, T, S).
        dtype: Optional output dtype; defaults to the dtype of s.

    Returns:
        Tensor of shape (B, H, T, S) with cumulative sums computed along dim T independently per S.
    """
    B, H, T, S = s.shape
    dtype = dtype or s.dtype
    z = torch.empty_like(s, dtype=dtype)
    for b in range(B):
        for h in range(H):
            for s_idx in range(S):
                z_bh = torch.cumsum(s[b, h, :, s_idx], dim=0)
                z[b, h, :, s_idx] = z_bh
    return z
