import torch

ROPE_GROUP_SIZE = 4

def torch_fast_rope_embedding(Q: torch.Tensor, K: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Pure PyTorch version of Fast RoPE embedding.
    Q, K: [batch, n_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim//2]
    """
    batch, n_heads, seq_len, head_dim = Q.shape
    half_dim = head_dim // 2

    # reshape to [batch, seq_len, n_heads, head_dim] for easier broadcasting
    Q_ = Q.permute(0, 2, 1, 3)  # [batch, seq_len, n_heads, head_dim]
    K_ = K.permute(0, 2, 1, 3)

    # split head_dim
    Q1, Q2 = Q_[..., :half_dim], Q_[..., half_dim:]
    K1, K2 = K_[..., :half_dim], K_[..., half_dim:]

    # cos/sin: [seq_len, 1, half_dim] -> broadcast over batch and heads
    cos = cos[:, None, :].to(Q.dtype)
    sin = sin[:, None, :].to(Q.dtype)

    def apply_rope(x1, x2, cos, sin, backward=False):
        if backward:
            sin = -sin
        new_x1 = x1 * cos - x2 * sin
        new_x2 = x2 * cos + x1 * sin
        return torch.cat([new_x1, new_x2], dim=-1)

    Q_out = apply_rope(Q1, Q2, cos, sin)
    K_out = apply_rope(K1, K2, cos, sin)

    # reshape back to [batch, n_heads, seq_len, head_dim]
    Q_out = Q_out.permute(0, 2, 1, 3).contiguous()
    K_out = K_out.permute(0, 2, 1, 3).contiguous()

    return Q_out
