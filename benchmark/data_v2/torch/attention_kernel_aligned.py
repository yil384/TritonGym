import torch
import torch.nn.functional as F

def torch_attention_kernel_aligned(q, k, v, rel_h_w, sm_scale=1.0,
                                   BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=2):
    """
    q, k, v: [B, H, N_CTX, D]
    rel_h_w: [B, H, N_CTX, 128]   # relative position bias
    sm_scale: float
    BLOCK_M, BLOCK_N, num_warps, num_stages: ignored in torch impl (kernel tuning params)
    """
    B, H, N_CTX, D = q.shape
    device, dtype = q.device, q.dtype

    # Compute in fp32 for numerical stability to match Triton kernel accumulation
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    rel_h_w_f = rel_h_w.float()

    # ---- Step 1: compute qk in fp32 ----
    qk = torch.matmul(q_f, k_f.transpose(-1, -2))  # [B, H, N_CTX, N_CTX]
    qk = qk * (sm_scale * 1.44269504)          # Triton uses exp2

    # ---- Step 2: construct bias (matching b0+b1) ----
    bias_last_size = rel_h_w.size(-1) // 2  # =64

    # b0: for column j, index is j // BLOCK_N into rel_h_w last dim
    col_indices = torch.arange(N_CTX, device=device)
    b0_indices = col_indices // BLOCK_N
    b0 = rel_h_w_f[..., b0_indices]  # [B, H, N_CTX, N_CTX]

    # b1: indices (arange(BLOCK_N) % bias_last_size) + bias_last_size, tiled
    b1_col_indices = (torch.arange(BLOCK_N, device=device) % bias_last_size) + bias_last_size
    b1_block = rel_h_w_f[..., b1_col_indices]  # [B, H, N_CTX, BLOCK_N]
    n_repeats = (N_CTX + BLOCK_N - 1) // BLOCK_N
    b1 = b1_block.repeat(1, 1, 1, n_repeats)[..., :N_CTX]  # [B, H, N_CTX, N_CTX]

    qk = qk + (b0 + b1)

    # ---- Step 3: numerically stable softmax in fp32 ----
    m_i = qk.max(dim=-1, keepdim=True).values     # [B,H,N_CTX,1]
    p = torch.exp2(qk - m_i)                      # exp2 instead of exp
    l_i = p.sum(dim=-1, keepdim=True)             # normalization factor
    attn = p / l_i

    # ---- Step 4: apply V ----
    out = torch.matmul(attn.to(dtype), v)
    return out
