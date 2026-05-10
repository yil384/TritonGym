import torch
import torch.nn.functional as F

def torch_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal=True, sm_scale=1.0):
    """
    PyTorch version of Flash Attention (small sequence version, simulating Triton kernel logic).

    Args:
        q, k, v: [batch, heads, seq_len, dim] input tensors.
        causal: Whether to use a causal mask.
        sm_scale: Softmax scaling factor.

    Returns:
        out: [batch, heads, seq_len, dim] attention output.
    """
    B, H, L, D = q.shape
    # Scale by sm_scale only (F.softmax uses natural exp, not exp2)
    q_scaled = q * sm_scale

    # Compute attention scores
    # [B,H,L,D] @ [B,H,D,L] -> [B,H,L,L]
    attn_scores = torch.matmul(q_scaled, k.transpose(-2, -1).to(q.dtype))

    if causal:
        mask = torch.tril(torch.ones((L, L), device=q.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask[None, None, :, :], float('-inf'))

    attn_probs = F.softmax(attn_scores, dim=-1)

    out = torch.matmul(attn_probs, v)
    return out
