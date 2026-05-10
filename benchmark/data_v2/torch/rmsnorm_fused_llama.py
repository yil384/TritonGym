import torch

def torch_rmsnorm_fused_llama(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """
    Fused RMSNorm forward in PyTorch (CPU/GPU)
    Args:
        x: [batch, feature_dim]
        weight: [feature_dim]
        eps: epsilon for numerical stability
    Returns:
        y: [batch, feature_dim]
    """
    # Compute in float32 for precision (matches Triton kernel which casts to tl.float32)
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + eps)
    x_hat = x_f / rms
    # Apply weight and cast to float16 (Triton stores as tl.float16)
    y = (x_hat * weight.float()).to(torch.float16)
    return y
