import torch
import math

def torch_attention_llama(Q, K, V, sm_scale=0.1, is_causal=False, start_position=0):
    """Compute scaled dot-product attention as used in LLaMA models."""
    sm_scale = 1.0 / math.sqrt(Q.shape[-1])
    # Compute in fp32 for numerical stability
    attn_scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * sm_scale
    if is_causal:
        seq_len = Q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V.float())
    return output.to(Q.dtype)
