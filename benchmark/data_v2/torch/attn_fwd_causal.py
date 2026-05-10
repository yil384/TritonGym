import torch

def torch_attn_fwd_causal(q, k, v):
    """Compute causal scaled dot-product attention with per-position scaling.

    Args:
        q: Query tensor of shape (batch, head, seq_len, head_dim).
        k: Key tensor of shape (batch, head, seq_len, head_dim).
        v: Value tensor of shape (batch, head, seq_len, head_dim).

    Returns:
        Attention output tensor of shape (batch, head, seq_len, head_dim).
    """
    batch_size, num_heads, seq_length, head_dim = q.shape
    q_scale = torch.ones((batch_size, num_heads, seq_length), device=q.device, dtype=q.dtype)
    k_scale = torch.ones((batch_size, num_heads, seq_length), device=k.device, dtype=k.dtype)
    attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * q_scale.unsqueeze(-1) * k_scale.unsqueeze(-2)
    mask = torch.tril(torch.ones(seq_length, seq_length, device=q.device)).unsqueeze(0).unsqueeze(0)
    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(attn_scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    return out
