import torch

def torch_attn_relu(q, k, v, causal=True, sm_scale=1, warp_specialize=False):
    """Compute attention with ReLU activation instead of softmax on attention scores.

    Args:
        q: Query tensor of shape (B, num_heads, seq_len, head_dim).
        k: Key tensor of shape (B, num_heads, seq_len, head_dim).
        v: Value tensor of shape (B, num_heads, seq_len, head_dim).
        causal: If True, apply a causal mask (masked positions set to -inf before ReLU).
        sm_scale: Scale factor applied to QK^T scores.
        warp_specialize: Unused Triton tuning parameter.

    Returns:
        Attention output tensor of shape (B, num_heads, seq_len, head_dim).
    """
    M = torch.tril(torch.ones((q.shape[2], k.shape[2]), device=q.device))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.relu(p)
    # p = torch.softmax(p.float(), dim=-1)
    p = p.to(q.dtype)
    # p = torch.exp(p)
    out = torch.matmul(p, v)
    return out
