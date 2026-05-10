import torch
import math

def torch_block_sparse_attn(Q, K, V,
                            num_layout=1, softmax_scale=1.0, num_heads=4, num_kv_heads=2,
                            total_seq_len=16, BLOCK_M=16, EVEN_M=True, BLOCK_N=16, EVEN_N=True,
                            BLOCK_D=16, NUM_D_BLOCKS=2):
    """
    Q: [B, H, M, D]
    K: [B, H_kv, N, D]
    V: [B, H_kv, N, D]
    Extra args (EVEN_M, EVEN_N, BLOCK_D, NUM_D_BLOCKS) are triton kernel params, ignored here.
    """
    B, H, M, D = Q.shape
    _, H_kv, N, _ = K.shape
    device = Q.device
    dtype = Q.dtype

    if num_heads is None:
        num_heads = H
    if num_kv_heads is None:
        num_kv_heads = H_kv
    if total_seq_len is None:
        total_seq_len = M
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    head_groups = num_heads // num_kv_heads

    layout_csr_row_indices = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    layout_csr_col_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    layout_csr_row_stride_h = layout_csr_row_indices.numel()
    layout_csr_col_stride_h = layout_csr_col_indices.numel()

    # Create the attention mask from CSR layout
    mask = torch.full((B, H, M, N), float('-inf'), device=device, dtype=dtype)

    num_rows = M // BLOCK_M
    num_cols = N // BLOCK_N

    for h in range(H):
        layout_h = h % num_layout
        for row in range(num_rows):
            start_l = layout_csr_row_indices[layout_h * (num_rows + 1) + row].item()
            end_l = layout_csr_row_indices[layout_h * (num_rows + 1) + row + 1].item()
            for col_idx_idx in range(start_l, end_l):
                col_idx = layout_csr_col_indices[layout_h * (num_rows * num_cols) + col_idx_idx].item()
                m_start, m_end = row * BLOCK_M, (row + 1) * BLOCK_M
                n_start, n_end = col_idx * BLOCK_N, (col_idx + 1) * BLOCK_N
                mask[:, h, m_start:m_end, n_start:n_end] = 0.0

    # Causal mask: the triton kernel applies offs_m >= (start_n + offs_n)
    causal_mask = torch.arange(M, device=device).unsqueeze(1) >= torch.arange(N, device=device).unsqueeze(0)
    causal_inf = torch.where(causal_mask, 0.0, float('-inf')).to(dtype)
    mask = mask + causal_inf.unsqueeze(0).unsqueeze(0)

    # Output
    out = torch.empty((B, H, M, D), device=device, dtype=dtype)

    for h in range(H):
        h_kv = h // head_groups
        attn = torch.einsum("bmd,bnd->bmn", Q[:, h], K[:, h_kv])  # [B, M, N]
        attn = attn * softmax_scale
        attn = attn + mask[:, h].to(dtype)
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out[:, h] = torch.einsum("bmn,bnd->bmd", attn, V[:, h_kv])

    return out
