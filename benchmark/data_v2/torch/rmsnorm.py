import torch

def torch_rmsnorm(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm implementation, equivalent to the Triton version.

    Args:
        x: Input tensor, shape (batch, M, K).
        rms_w: Weight tensor, shape (K,).
        eps: Epsilon to avoid division by zero.

    Returns:
        Output tensor, shape (batch, M, K).
    """
    batch, M, K = x.shape
    assert rms_w.shape[0] == K

    # Compute RMS
    var = (x.float() ** 2).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)

    # Apply weights
    out = x.float() * rstd * rms_w.view(1, 1, K)
    return out.to(x.dtype)
