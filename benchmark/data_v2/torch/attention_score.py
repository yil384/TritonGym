import torch
import math

def torch_attention_score(q, k, sliding_window=None, complement_sliding_window=False):
    """
    PyTorch equivalent implementation for verifying the Triton version.

    Args:
        q: (B, H, N_CTX, D)
        k: (B, H, N_KV_CTX, D)
        sliding_window: (offset, size) or None
        complement_sliding_window: bool

    Returns:
        out: (B, H, N_KV_CTX)
    """
    B, H, N_CTX, D = q.shape
    N_KV_CTX = k.size(2)
    sm_scale = 1.0 / math.sqrt(D)

    # (B, H, N_CTX, N_KV_CTX)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Convert to log base 2 as in the Triton version, since it uses exp2
    attn_scores = attn_scores * 1.4426950408889634  # 1 / log(2)

    # In Triton, M is the shift amount from max(QK), but in the test code M is initialized to 0,
    # so the equivalent PyTorch version does not subtract.
    # If FlashAttention's cumulative M logic is added later, subtraction would be needed here.

    # (B, H, N_CTX, N_KV_CTX) -> element-wise exp2
    p = torch.exp2(attn_scores)

    # Handle sliding window
    if sliding_window is not None:
        offset, size = sliding_window
        # Build a (N_CTX, N_KV_CTX) distance matrix
        ctx_ids = torch.arange(N_CTX, device=q.device).unsqueeze(1)  # (N_CTX, 1)
        kv_ids = torch.arange(N_KV_CTX, device=q.device).unsqueeze(0)  # (1, N_KV_CTX)
        dist = ctx_ids - kv_ids + offset  # (N_CTX, N_KV_CTX)

        if complement_sliding_window:
            mask = (dist >= size)
        else:
            mask = (dist >= 0) & (dist < size)

        # Apply mask
        p = p * mask.to(p.dtype)

    # In Triton the final step is o += sum(p, axis=0), so here we sum along the seq_len dimension
    # (B, H, N_CTX, N_KV_CTX) -> (B, H, N_KV_CTX)
    out = p.sum(dim=2)

    return out
