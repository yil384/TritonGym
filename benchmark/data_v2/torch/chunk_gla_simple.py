import torch

def torch_chunk_gla_simple(h, q, k, v, g, BT=32, scale=0.1):
    """
    PyTorch implementation matching Triton chunk_simple_gla_fwd_kernel_o.
    h: [B, H, K, V] (or [B, H, NT*K, V] for multi-chunk, but here single chunk state)
    q: [B, H, T, K]
    k: [B, H, T, K]
    v: [B, H, T, V]
    g: [B, H, T]
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    o = torch.zeros_like(v)
    NT = (T + BT - 1) // BT

    for b in range(B):
        for hh in range(H):
            for nt in range(NT):
                t_start = nt * BT
                t_end = min(t_start + BT, T)
                chunk_len = t_end - t_start

                q_chunk = q[b, hh, t_start:t_end, :]   # [chunk_len, K]
                # k is [B,H,T,K]; triton loads it as (K,T) via transposed strides
                k_chunk = k[b, hh, t_start:t_end, :]   # [chunk_len, K]
                v_chunk = v[b, hh, t_start:t_end, :]   # [chunk_len, V]
                g_chunk = g[b, hh, t_start:t_end]       # [chunk_len]

                # h state for this chunk: h[b,hh, nt*K:(nt+1)*K, :]
                # If h is [B,H,K,V] (single chunk), use h directly
                if h.shape[2] == K:
                    h_chunk = h[b, hh, :, :]            # [K, V]
                else:
                    h_chunk = h[b, hh, nt*K:(nt+1)*K, :]  # [K, V]

                # b_o = q @ h  (inter-chunk)
                b_o = torch.matmul(q_chunk, h_chunk).float()  # [chunk_len, V]
                # b_s = q @ k^T (intra-chunk)
                b_s = torch.matmul(q_chunk, k_chunk.t()).float()  # [chunk_len, chunk_len]

                # Apply gating
                b_o = b_o * torch.exp(g_chunk).unsqueeze(-1)
                b_s = b_s * torch.exp(g_chunk[:, None] - g_chunk[None, :])

                # Causal mask
                o_i = torch.arange(chunk_len, device=q.device)
                m_s = o_i[:, None] >= o_i[None, :]
                b_s = torch.where(m_s, b_s, torch.zeros_like(b_s))

                b_o = (b_o + torch.matmul(b_s.to(v_chunk.dtype), v_chunk)) * scale
                o[b, hh, t_start:t_end, :] = b_o

    return o
