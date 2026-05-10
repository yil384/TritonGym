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
    Pure PyTorch implementation of mixed sparse attention (approximation of Triton kernel).
    Only core computation, supports batch, heads, causal masking, and sparse blocks.
    """
    B, H, N_CTX, D_HEAD = q.shape
    o = torch.zeros_like(q, dtype=q.dtype, device=q.device)

    seqlens = torch.full((B,), N_CTX, dtype=torch.int32, device=q.device)
    block_count = torch.ones((B, H, N_CTX // block_size_M), dtype=torch.int32, device=q.device)
    block_offset = torch.zeros((B, H, N_CTX // block_size_M, 4), dtype=torch.int32, device=q.device)
    column_count = torch.ones((B, H, N_CTX // block_size_M), dtype=torch.int32, device=q.device)
    column_index = torch.zeros((B, H, N_CTX // block_size_M, 8), dtype=torch.int32, device=q.device)
    
    qk_scale = sm_scale * 1.44269504  # Triton uses * ln2

    for b in range(B):
        for h in range(H):
            seqlen = seqlens[b].item()
            q_bh = q[b, h, :seqlen] * qk_scale
            k_bh = k[b, h, :seqlen]
            v_bh = v[b, h, :seqlen]

            acc = torch.zeros_like(q_bh)
            l_i = torch.zeros(seqlen, device=q.device, dtype=q.dtype)
            m_i = torch.full((seqlen,), float('-inf'), device=q.device, dtype=q.dtype)

            # Sparse block forward
            for blk_idx in range(block_count[b, h].numel()):
                start_m = blk_idx * block_size_M
                if start_m >= seqlen:
                    continue
                end_m = min(start_m + block_size_M, seqlen)
                q_block = q_bh[start_m:end_m]  # [M, D]

                for n_idx in range(column_count[b, h, blk_idx].item()):
                    start_n = column_index[b, h, blk_idx, n_idx].item()
                    end_n = min(start_n + block_size_N, seqlen)
                    k_block = k_bh[start_n:end_n]  # [N, D]
                    v_block = v_bh[start_n:end_n]  # [N, D]

                    # Compute attention scores
                    qk = torch.matmul(q_block, k_block.T)  # [M, N]
                    causal_mask = torch.ones_like(qk, dtype=torch.bool)
                    for mi in range(qk.shape[0]):
                        for ni in range(qk.shape[1]):
                            if start_n + ni > start_m + mi:
                                causal_mask[mi, ni] = False
                    qk = torch.where(causal_mask, qk, torch.tensor(float('-inf'), device=q.device))

                    m_i_new = torch.maximum(m_i[start_m:end_m], torch.max(qk, dim=1).values)
                    alpha = torch.pow(2, m_i[start_m:end_m] - m_i_new)
                    p = torch.pow(2, qk - m_i_new[:, None])
                    acc[start_m:end_m] = acc[start_m:end_m] * alpha[:, None] + torch.matmul(p, v_block)
                    l_i[start_m:end_m] = l_i[start_m:end_m] * alpha + torch.sum(p, dim=1)
                    m_i[start_m:end_m] = m_i_new

            # Normalize
            o[b, h, :seqlen] = acc / l_i[:, None]

    return o
