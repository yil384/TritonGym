import torch
import torch.nn.functional as F
from typing import Optional

def torch_tlx_hstu_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Pure PyTorch HSTU attention (no fbgemm dependency).
    q, k: [L, H, D]
    v: [L, H, V]
    Returns: [L, H, V]
    """
    L, H, D = q.shape
    V = v.shape[2]

    # Treat as single batch: [1, H, L, D]
    q_4d = q.permute(1, 0, 2).unsqueeze(0)  # [1, H, L, D]
    k_4d = k.permute(1, 0, 2).unsqueeze(0)  # [1, H, L, D]
    v_4d = v.permute(1, 0, 2).unsqueeze(0)  # [1, H, L, V]

    # Attention scores
    qk_attn = torch.matmul(q_4d, k_4d.transpose(-2, -1)) * alpha  # [1, H, L, L]

    # SiLU activation + normalization (matches HSTU)
    qk_attn = F.silu(qk_attn) / L

    # Causal mask
    causal_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
    qk_attn = qk_attn * causal_mask.unsqueeze(0).unsqueeze(0)

    # Attention output
    attn_out = torch.matmul(qk_attn, v_4d)  # [1, H, L, V]

    # Back to [L, H, V]
    return attn_out.squeeze(0).permute(1, 0, 2)
