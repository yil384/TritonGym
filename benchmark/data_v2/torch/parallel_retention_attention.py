import torch

def torch_parallel_retention_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Simplified PyTorch version of Triton parallel retention attention.
    This is the core forward computation only.
    Args:
        q: [B, H, T, D_qk]
        k: [B, H, T, D_qk]
        v: [B, H, T, D_v]
    Returns:
        o: [B, H, T, D_v]
    """
    B, H, T, D_qk = q.shape
    D_v = v.shape[-1]
    scale = D_qk ** -0.5

    # Decay rates per head
    b_b = torch.log2(1 - torch.exp2(-5 - torch.arange(H, device=q.device, dtype=q.dtype)))

    # Initialize output
    o = torch.zeros(B, H, T, D_v, device=q.device, dtype=q.dtype)

    for b in range(B):
        for h in range(H):
            decay = 2 ** (torch.arange(T, device=q.device, dtype=q.dtype) * b_b[h])
            for t in range(T):
                # Cumulative retention attention
                weights = torch.zeros(T, device=q.device, dtype=q.dtype)
                for s in range(t + 1):
                    weights[s] = (q[b, h, t] @ k[b, h, s] * scale) * (2 ** ((t - s) * b_b[h]))
                o[b, h, t] = torch.sum(weights.unsqueeze(-1) * v[b, h, :T], dim=0)
    return o
