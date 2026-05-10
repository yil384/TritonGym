import torch

def torch_attention_forward(q, k, v):
    """
    q, k, v: [B, H, L, D]
    Standard attention without 1/sqrt(d) scaling (q_scale=k_scale=1).
    Uses exp2-based softmax matching Triton kernel.
    """
    B, H, L, D = q.shape

    # Compute attention scores in fp32
    attn_scores = torch.matmul(q.float(), k.float().transpose(-1, -2))

    # Softmax (exp2-based is mathematically equivalent to regular softmax)
    attn_scores = attn_scores - attn_scores.amax(dim=-1, keepdim=True)
    attn_weights = torch.exp(attn_scores)
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    # Output
    out = torch.matmul(attn_weights, v.float())
    return out.to(torch.float16)
