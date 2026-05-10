import torch

def torch_parallel_attention(q, k, v, scale=None, eps=1e-5, use_normalize=True, return_both=False):
    """
    Pure PyTorch implementation of the Triton parallel-rebased attention.
    Only core forward computation.
    
    q: [B, H, T, K]
    k: [B, H, T, K]
    v: [B, H, T, V]
    scale: float or None
    """
    B, H, T, K = q.shape
    _, _, _, V = v.shape

    if scale is None:
        scale = K ** -0.5

    # Compute attention scores
    # [B, H, T, T]
    attn_scores = torch.matmul(q * scale, k.transpose(-2, -1))  # scaled dot-product

    # Square the scores as in Triton kernel
    attn_scores = attn_scores ** 2

    # Apply causal mask (lower triangular, including diagonal)
    causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    attn_scores = attn_scores.masked_fill(~causal_mask[None, None, :, :], 0.0)

    # Sum along sequence for normalization
    z = attn_scores.sum(dim=-1)  # [B, H, T]

    # Weighted sum with V
    o = torch.matmul(attn_scores, v)  # [B, H, T, V]

    if use_normalize:
        o = o / (z[..., None] + eps)

    if return_both:
        return o, z
    return o
