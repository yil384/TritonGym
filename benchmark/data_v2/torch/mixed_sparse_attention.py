import torch

def torch_mixed_sparse_attention(
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    sm_scale: float = 0.1,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of the MInference mixed (block + vertical column) sparse
    attention used by the Triton kernel, with the same fixed sparse pattern as the Triton wrapper.

    For every row block of block_size_M queries (b, h, r):
      - block part : keys in [block_offset[b,h,r,i], block_offset[b,h,r,i] + block_size_N)
                     for i < block_count[b,h,r], restricted to keys < seqlen, causally masked;
      - column part: the individual keys column_index[b,h,r,:column_count[b,h,r]] (no causal mask).
    All selected (row, key) scores enter one softmax (a key selected by both parts counts twice,
    exactly like the kernel). Rows >= seqlen are left zero.
    """
    B, H, N_CTX, D_HEAD = q.shape
    o = torch.zeros_like(q, dtype=q.dtype, device=q.device)

    num_rows = (N_CTX + block_size_M - 1) // block_size_M
    seqlens = torch.full((B,), N_CTX, dtype=torch.int32, device=q.device)
    block_count = torch.ones((B, H, num_rows), dtype=torch.int32, device=q.device)
    block_offset = torch.zeros((B, H, num_rows, 4), dtype=torch.int32, device=q.device)
    column_count = torch.ones((B, H, num_rows), dtype=torch.int32, device=q.device)
    column_index = torch.zeros((B, H, num_rows, 8), dtype=torch.int32, device=q.device)

    for b in range(B):
        seqlen = int(seqlens[b].item())
        for h in range(H):
            qf = q[b, h].float() * sm_scale
            kf = k[b, h].float()
            vf = v[b, h].float()
            for r in range(num_rows):
                start_m = r * block_size_M
                if start_m >= seqlen:
                    continue
                end_m = min(start_m + block_size_M, seqlen)
                rows = torch.arange(start_m, end_m, device=q.device)
                q_blk = qf[start_m:end_m]  # [M, D]

                scores, values = [], []
                # block-sparse part (causal)
                for i in range(int(block_count[b, h, r].item())):
                    start_n = int(block_offset[b, h, r, i].item())
                    end_n = min(start_n + block_size_N, seqlen)
                    if end_n <= start_n:
                        continue
                    cols = torch.arange(start_n, end_n, device=q.device)
                    s = q_blk @ kf[start_n:end_n].T  # [M, N]
                    s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
                    scores.append(s)
                    values.append(vf[start_n:end_n])
                # vertical column part (no causal mask)
                n_cols = int(column_count[b, h, r].item())
                if n_cols > 0:
                    cols = column_index[b, h, r, :n_cols].long()
                    scores.append(q_blk @ kf[cols].T)
                    values.append(vf[cols])

                s = torch.cat(scores, dim=1)
                p = torch.softmax(s, dim=1)
                o[b, h, start_m:end_m] = (p @ torch.cat(values, dim=0)).to(q.dtype)

    return o
