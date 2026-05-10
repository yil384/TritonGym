import torch

def torch_attn_without_softmax(q, k, v, causal=True, sm_scale=1, warp_specialize=False):
    """Compute linear attention without softmax normalization.

    Computes raw QK^T scaled scores with optional causal masking, then multiplies by V.

    Args:
        q: Query tensor of shape (B, num_heads, seq_len, head_dim).
        k: Key tensor of shape (B, num_heads, seq_len, head_dim).
        v: Value tensor of shape (B, num_heads, seq_len, head_dim).
        causal: If True, zero out above-diagonal attention scores.
        sm_scale: Scale factor applied to QK^T scores.
        warp_specialize: Unused Triton tuning parameter.

    Returns:
        Attention output tensor of shape (B, num_heads, seq_len, head_dim).
    """
    p = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        M = torch.tril(torch.ones((q.shape[2], k.shape[2]), device=q.device))
        p[:, :, M == 0] = 0.0
    out = torch.matmul(p.to(v.dtype), v)
    return out
