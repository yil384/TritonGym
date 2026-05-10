import torch
from typing import Tuple

def torch_rms_matmul_rbe(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    rms_w: torch.Tensor,
    n_heads: int = 32,
    head_dim: int = 128,
    eps: float = 1e-6,
    theta: float = 10000.
) -> torch.Tensor:
    """
    Pure PyTorch implementation of RMS-normalized QKV matmul.

    Args:
        x: input tensor of shape (batch, seq_len, embed_dim)
        q_weight/k_weight/v_weight: weight matrices (embed_dim, embed_dim)
        rms_w: RMS weights (embed_dim,)
        n_heads: number of attention heads
        head_dim: dimension per head
        eps: epsilon for numerical stability
        theta: frequency scaling factor for rotary embeddings (unused in current impl)

    Returns:
        q tensor of shape (batch, seq_len, n_heads, head_dim)
    """
    batch, seq_len, embed_dim = x.shape
    assert embed_dim == n_heads * head_dim

    # RMS normalization: sqrt(mean(x^2) + eps), then x / rms * rms_w
    # The triton kernel computes: sum(x^2) / K + eps, then rsqrt
    # mean = sum(x^2) / K, so this is equivalent
    rms = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x.float() / rms * rms_w.float()

    # Linear projection
    q = (x_norm @ q_weight.t().float()).to(torch.float16)
    k = (x_norm @ k_weight.t().float()).to(torch.float16)
    v = (x_norm @ v_weight.t().float()).to(torch.float16)

    # Reshape to (batch, seq_len, n_heads, head_dim)
    q = q.view(batch, seq_len, n_heads, head_dim)
    k = k.view(batch, seq_len, n_heads, head_dim)
    v = v.view(batch, seq_len, n_heads, head_dim)

    return q
