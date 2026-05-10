import torch

def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    RMS normalization in pure PyTorch.

    Args:
        x: input tensor of shape (..., normalized_shape)
        normalized_shape: tuple of dimensions to normalize over
        weight: scaling weights of shape `normalized_shape`
        eps: epsilon for numerical stability

    Returns:
        normalized tensor of same shape as x
    """
    # Compute RMS along the last `len(normalized_shape)` dimensions
    normalized_shape = weight.shape
    dim = tuple(range(-len(normalized_shape), 0))
    rms = torch.sqrt(torch.mean(x * x, dim=dim, keepdim=True) + eps)

    # Normalize and scale
    y = x / rms * weight

    return y
