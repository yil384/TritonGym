import torch

def torch_attn_gelu(q, k, v, causal=True, sm_scale=1.0, warp_specialize=False):
    """Compute attention with GELU activation instead of softmax on attention scores.

    Args:
        q: Query tensor of shape (B, num_heads, seq_len, head_dim).
        k: Key tensor of shape (B, num_heads, seq_len, head_dim).
        v: Value tensor of shape (B, num_heads, seq_len, head_dim).
        causal: If True, apply a causal (lower-triangular) mask.
        sm_scale: Scale factor applied to QK^T scores.
        warp_specialize: Unused Triton tuning parameter.

    Returns:
        Attention output tensor of shape (B, num_heads, seq_len, head_dim).
    """
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    p = torch.matmul(q_f, k_f.transpose(2, 3)) * sm_scale
    if causal:
        M = torch.tril(torch.ones((q.shape[2], q.shape[2]), device=q.device))
        p[:, :, M == 0] = -1e9
    # Standard GELU in float32
    p = p * 0.5 * (1.0 + torch.erf(p / 1.4142135623730951))
    p = p.to(q.dtype)
    out = torch.matmul(p.float(), v_f)
    return out.to(q.dtype)
