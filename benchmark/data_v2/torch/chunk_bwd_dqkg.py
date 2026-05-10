import torch

def torch_chunk_bwd_dqkg(do, q, k, v, g, h, dh, scale=0.1):
    """
    PyTorch implementation matching Triton chunk_simple_gla_bwd_kernel_dqkg.
    Processes in chunks of BT=64 along the T dimension.

    Inputs:
        do: [B, H, T, V]
        q:  [B, H, T, K]
        k:  [B, H, T, K]
        v:  [B, H, T, V]
        g:  [B, H, T]
        h:  [B, H, NT*K, V]  (chunked hidden state, NT = ceil(T/BT))
        dh: [B, H, NT*K, V]
        scale: float
    Output:
        dq: [B, H, T, K]
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    BT = 64
    NT = (T + BT - 1) // BT

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)

    for b in range(B):
        for hh in range(H):
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                chunk_len = t_end - t_start

                q_blk = q[b, hh, t_start:t_end, :]        # [chunk_len, K]
                k_blk = k[b, hh, t_start:t_end, :]        # [chunk_len, K]
                v_blk = v[b, hh, t_start:t_end, :]        # [chunk_len, V]
                do_blk = do[b, hh, t_start:t_end, :]      # [chunk_len, V]
                g_blk = g[b, hh, t_start:t_end]           # [chunk_len]
                h_blk = h[b, hh, i_t*K:(i_t+1)*K, :]       # [K, V]
                dh_blk = dh[b, hh, i_t*K:(i_t+1)*K, :]    # [K, V]

                g_last = g_blk[-1]

                # Accumulate over V dimension
                # h_blk is [K,V], triton reads it as (V,K) via block ptr -> do @ h^T
                b_dq = torch.matmul(do_blk, h_blk.t())     # [chunk_len, V] @ [V, K] = [chunk_len, K]
                # dh_blk is [K,V], triton reads it as (V,K) -> v @ dh^T
                b_dk = torch.matmul(v_blk, dh_blk.t())     # [chunk_len, V] @ [V, K] = [chunk_len, K]
                b_ds = torch.matmul(do_blk, v_blk.t())     # [chunk_len, chunk_len]
                b_dg_last = (h_blk * dh_blk).sum()

                # Apply gating
                b_dg_last = b_dg_last * torch.exp(g_last)
                b_dq = b_dq * torch.exp(g_blk).unsqueeze(-1) * scale
                b_dk = b_dk * torch.exp(-g_blk + g_last).unsqueeze(-1)
                b_dg_last = b_dg_last + (b_dk * k_blk).sum()

                # Lower-triangular causal mask for intra-chunk attention
                o_i = torch.arange(chunk_len, device=q.device)
                mask = o_i[:, None] >= o_i[None, :]
                b_ds = torch.where(mask, b_ds * scale * torch.exp(g_blk[:, None] - g_blk[None, :]), torch.zeros_like(b_ds))

                b_dq = b_dq + torch.matmul(b_ds, k_blk)
                b_dk = b_dk + torch.matmul(b_ds.t(), q_blk)

                b_dg = (q_blk * b_dq - k_blk * b_dk).sum(-1)
                # Add b_dg_last to the last position in the chunk
                b_dg[-1] = b_dg[-1] + b_dg_last

                dq[b, hh, t_start:t_end, :] = b_dq
                dk[b, hh, t_start:t_end, :] = b_dk

    return dq
