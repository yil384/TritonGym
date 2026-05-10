import torch

def torch_rope_embedding_forward(Q, cos, sin):
    """
    PyTorch RoPE forward pass.
    Q: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim // 2) or with extra dims
    """
    Q = Q.transpose(1, 2).clone()  # (batch, seq_len, n_heads, head_dim)
    cos, sin = cos.squeeze(), sin.squeeze()
    batch, seq_len, n_heads, head_dim = Q.shape
    half_dim = head_dim // 2

    # Compute in fp32 for precision
    Q_f = Q.float()
    cos_ = cos.float().view(1, seq_len, 1, half_dim)
    sin_ = sin.float().view(1, seq_len, 1, half_dim)

    Q1 = Q_f[..., :half_dim]
    Q2 = Q_f[..., half_dim:]

    Q1_new = Q1 * cos_ - Q2 * sin_
    Q2_new = Q2 * cos_ + Q1 * sin_

    Q_out = torch.cat([Q1_new, Q2_new], dim=-1)
    Q_out = Q_out.to(Q.dtype)
    Q_out = Q_out.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
    return Q_out
