import torch

def torch_bmm_chunk_bwd(a, dout, residual=None, out=None):
    """
    Torch reference for Triton bmm_chunk_bwd kernel.

    a: [batch, seqlen, k] or [batch, seqlen, ngroups, k]
    dout: [batch, nchunks, chunk_size, chunk_size] or [batch, nchunks, ngroups, chunk_size, chunk_size]
    residual: same shape as a (optional, added to result)
    out: pre-allocated output (optional)

    The kernel computes for each chunk c:
        db[start:end] = dout[b,c].T @ a[b, start:end]
    where dout.T is used because of the stride-based transposition in the kernel.
    """
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
        nchunks = dout.shape[1]
        chunk_size = dout.shape[-1]
    else:
        batch, seqlen, ngroups, k = a.shape
        nchunks = dout.shape[1]
        chunk_size = dout.shape[-1]

    if out is not None:
        db = out
    else:
        db = torch.empty_like(a)

    for b_idx in range(batch):
        for c in range(nchunks):
            start = c * chunk_size
            end = min(start + chunk_size, seqlen)
            actual_cs = end - start

            if not has_groups:
                # dout[b,c]: [chunk_size, chunk_size], transposed in kernel
                # a_chunk: [actual_cs, k]
                # result: dout.T @ a = [chunk_size, chunk_size].T @ [actual_cs, k] = [chunk_size, k]
                dout_block = dout[b_idx, c, :actual_cs, :actual_cs]  # [actual_cs, actual_cs]
                a_chunk = a[b_idx, start:end, :]  # [actual_cs, k]
                db[b_idx, start:end, :] = torch.matmul(dout_block.float().T, a_chunk.float()).to(a.dtype)
            else:
                for g in range(ngroups):
                    dout_block = dout[b_idx, c, g, :actual_cs, :actual_cs]
                    a_chunk = a[b_idx, start:end, g, :]
                    db[b_idx, start:end, g, :] = torch.matmul(dout_block.float().T, a_chunk.float()).to(a.dtype)

    if residual is not None:
        db = db + residual

    return db
