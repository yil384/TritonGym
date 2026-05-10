import torch
from typing import Optional, Tuple


def torch_chunked_cumsum_fwd(
    dt: torch.Tensor,
    A: torch.Tensor,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = True,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
):
    """
    Torch reference implementation of chunked cumulative sum forward.
    Equivalent to triton_chunked_cumsum_fwd but written in PyTorch.

    Args:
        dt: (batch, seqlen, nheads)
        A: (nheads,)
        chunk_size: int
        dt_bias: (nheads,) optional
        dt_softplus: bool
        dt_limit: (min, max)

    Returns:
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        dt_out: (batch, nheads, nchunks, chunk_size)
    """
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    chunk_size = dt.shape[2]  # Assume chunk_size is the last dimension
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    # [B, H, Nchunks, Csize]
    dt_out = torch.zeros(batch, nheads, nchunks, chunk_size, dtype=torch.float32, device=dt.device)
    dA_cumsum = torch.zeros(batch, nheads, nchunks, chunk_size, dtype=torch.float32, device=dt.device)

    # Convert to float32
    dt = dt.to(torch.float32)
    if dt_bias is not None:
        dt = dt + dt_bias.view(1, 1, nheads)
    if dt_softplus:
        dt = torch.where(dt <= 20.0, torch.log1p(torch.exp(dt)), dt)
    dt = torch.clamp(dt, dt_limit[0], dt_limit[1])

    # Iterate over chunks
    for c in range(nchunks):
        start = c * chunk_size
        end = min((c + 1) * chunk_size, seqlen)
        clen = end - start

        dt_chunk = dt[:, start:end, :]            # [B, Clen, H]
        dt_chunk = dt_chunk.permute(0, 2, 1)      # [B, H, Clen]

        # Save to dt_out
        dt_out[:, :, c, :clen] = dt_chunk

        # dA = dt * A
        dA = dt_chunk * A.view(1, -1, 1)          # [B, H, Clen]
        dA_cumsum[:, :, c, :clen] = torch.cumsum(dA, dim=-1)

    return dA_cumsum
