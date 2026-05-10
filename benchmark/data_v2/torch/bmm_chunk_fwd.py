import torch
import math

def torch_bmm_chunk_fwd(a, b, chunk_size=32, seq_idx=None, causal=False, output_dtype=None):
    """
    Torch reference for Triton bmm_chunk_fwd kernel.

    a, b: [batch, seqlen, k] or [batch, seqlen, ngroups, k]
    chunk_size: int
    seq_idx: [batch, seqlen] or None
    causal: bool
    output_dtype: torch.dtype or None

    For each chunk c, computes: out[b,c] = a_chunk @ b_chunk^T
    where a_chunk = a[b, c*cs:(c+1)*cs, :], b_chunk = b[b, c*cs:(c+1)*cs, :]
    """
    has_groups = a.dim() == 4
    batch = a.size(0)
    seqlen = a.size(1)
    k = a.size(-1)
    ngroups = a.size(2) if has_groups else 1
    nchunks = math.ceil(seqlen / chunk_size)

    out_dtype = a.dtype if output_dtype is None else output_dtype

    if not has_groups:
        out = torch.zeros(batch, nchunks, chunk_size, chunk_size, device=a.device, dtype=out_dtype)
    else:
        out = torch.zeros(batch, nchunks, ngroups, chunk_size, chunk_size, device=a.device, dtype=out_dtype)

    for c in range(nchunks):
        start = c * chunk_size
        end = min((c + 1) * chunk_size, seqlen)
        actual_cs = end - start

        if not has_groups:
            a_chunk = a[:, start:end, :]  # [batch, actual_cs, K]
            b_chunk = b[:, start:end, :]  # [batch, actual_cs, K]
            # out[b,c, i, j] = sum_k a[b, start+i, k] * b[b, start+j, k]
            result = torch.matmul(a_chunk.float(), b_chunk.float().transpose(-1, -2))  # [batch, actual_cs, actual_cs]
            out[:, c, :actual_cs, :actual_cs] = result.to(out_dtype)
        else:
            for g in range(ngroups):
                a_chunk = a[:, start:end, g, :]
                b_chunk = b[:, start:end, g, :]
                result = torch.matmul(a_chunk.float(), b_chunk.float().transpose(-1, -2))
                out[:, c, g, :actual_cs, :actual_cs] = result.to(out_dtype)

    # Apply causal mask
    if causal:
        for i in range(chunk_size):
            for j in range(chunk_size):
                if j >= i + 1:
                    if not has_groups:
                        out[:, :, i, j] = 0.0
                    else:
                        out[:, :, :, i, j] = 0.0

    # Apply seq_idx mask
    if seq_idx is not None:
        for c in range(nchunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, seqlen)
            actual_cs = end - start
            seq_m = seq_idx[:, start:end]  # [batch, actual_cs]
            seq_n = seq_idx[:, start:end]  # [batch, actual_cs]
            mask = seq_m.unsqueeze(-1) == seq_n.unsqueeze(-2)  # [batch, actual_cs, actual_cs]
            if not has_groups:
                out[:, c, :actual_cs, :actual_cs] = out[:, c, :actual_cs, :actual_cs] * mask
            else:
                out[:, c, :, :actual_cs, :actual_cs] = out[:, c, :, :actual_cs, :actual_cs] * mask.unsqueeze(1)

    return out
