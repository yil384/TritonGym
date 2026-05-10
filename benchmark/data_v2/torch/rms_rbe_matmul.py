import torch

def torch_rms_rbe_matmul(x: torch.Tensor, weight: torch.Tensor, rms_w: torch.Tensor,
                          use_rbe: bool=True, start_pos: int=0, n_heads: int=8, head_dim: int=8, eps: float = 1e-6):
    """
    RMS normalization + (optional) rotary embedding + matrix multiplication
    Args:
        x: [batch, M, K]
        weight: [N, K]
        rms_w: [K]
        use_rbe: whether to apply rotary embeddings
        start_pos: starting token position for RBE
        n_heads: number of attention heads
        head_dim: dimension per head
        eps: epsilon for numerical stability
    Returns:
        out: [batch, M, n_heads, head_dim]
    """
    batch, M, K = x.shape
    N, K_W = weight.shape
    assert K == K_W

    # RMS normalization (compute in float32 to match triton)
    x_f32 = x.float()
    x_mean = torch.mean(x_f32 * x_f32, dim=2, keepdim=True) + eps
    x_norm_factor = torch.rsqrt(x_mean)  # [batch, M, 1]

    # x * rms_w in original dtype, then matmul accumulating in float32 (matching triton)
    x_scaled = x * rms_w  # element-wise multiply in original dtype
    out = torch.matmul(x_scaled.float(), weight.float().T)  # accumulate in float32
    out = (out * x_norm_factor).to(weight.dtype)  # apply norm and cast to output dtype

    # Note: The triton kernel declares RBE_EPILOGUE as a constexpr parameter
    # but never actually uses it in the kernel body, so RBE is not applied.
    # We skip rotary embedding to match the triton kernel behavior.

    out = out.view(batch, M, n_heads, head_dim)
    return out
