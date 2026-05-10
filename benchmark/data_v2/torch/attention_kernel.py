import torch

def torch_attention_kernel(q, k, v, rel_h_w, sm_scale=0.1,
                           BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2):
    """
    q, k, v: [B, H, N_CTX, D]
    rel_h_w: [B, H, N_CTX, 128]
    sm_scale: float
    """
    B, H, N_CTX, D = q.shape
    device, dtype = q.device, q.dtype

    qk_scale = sm_scale * 1.44269504

    # Compute in fp32
    q_f = q.float() * qk_scale
    k_f = k.float()
    v_f = v.float()

    bias_last_size = rel_h_w.size(-1) // 2  # = 64

    # Initialize accumulators
    m_i = torch.full((B, H, N_CTX), float('-inf'), device=device, dtype=torch.float32)
    l_i = torch.zeros((B, H, N_CTX), device=device, dtype=torch.float32)
    acc = torch.zeros((B, H, N_CTX, D), device=device, dtype=torch.float32)

    # Precompute b1 (same for all blocks of N)
    b1_col_indices = (torch.arange(BLOCK_N, device=device) % bias_last_size) + bias_last_size
    b1 = rel_h_w[..., b1_col_indices].float()  # [B, H, N_CTX, BLOCK_N]

    P_SEQ = 0
    hi = N_CTX + P_SEQ

    for start_n in range(0, hi, BLOCK_N):
        end_n = min(start_n + BLOCK_N, hi)
        # Load k, v block
        k_block = k_f[:, :, start_n:end_n, :]  # [B, H, BLOCK_N, D]
        v_block = v_f[:, :, start_n:end_n, :]

        # qk = dot(q_scaled, k^T)
        qk = torch.matmul(q_f, k_block.transpose(-1, -2))  # [B, H, N_CTX, BLOCK_N]

        # Bias: b0 = rel_h_w[..., start_n // BLOCK_N] (scalar per row)
        b0 = rel_h_w[..., start_n // BLOCK_N:start_n // BLOCK_N + 1].float()  # [B, H, N_CTX, 1]

        # Pad qk/b1 if block is partial
        actual_bn = end_n - start_n
        if actual_bn < BLOCK_N:
            # Pad to BLOCK_N
            pad = BLOCK_N - actual_bn
            qk = torch.nn.functional.pad(qk, (0, pad), value=float('-inf'))

        qk = qk + (b0 + b1) * 1.44269504

        # Trim back if padded
        if actual_bn < BLOCK_N:
            qk = qk[..., :actual_bn]
            v_block_use = v_block
        else:
            v_block_use = v_block

        # Online softmax update
        m_i_new = torch.maximum(m_i, qk.max(dim=-1).values)
        alpha = torch.pow(2.0, m_i - m_i_new)
        p = torch.pow(2.0, qk - m_i_new.unsqueeze(-1))

        acc = acc * alpha.unsqueeze(-1)
        acc = acc + torch.matmul(p.to(v_block_use.dtype), v_block_use)
        l_i = l_i * alpha + p.sum(dim=-1)
        m_i = m_i_new

    acc = acc / l_i.unsqueeze(-1)
    return acc.to(dtype)
