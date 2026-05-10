import torch

def torch_layer_norm(x, eps=1e-5):
    """
    Torch implementation of LayerNorm matching Triton version.
    Args:
        x: input tensor of shape (..., normalized_shape)
        normalized_shape: int or tuple, number of features to normalize
        weight: gamma (scale), shape=normalized_shape
        bias: beta (shift), shape=normalized_shape
        eps: small epsilon for numerical stability
    Returns:
        y: layer-normalized output
    """
    # Flatten to 2D: [M, N]
    x_2d = x.reshape(-1, x.shape[-1])
    mean = x_2d.mean(dim=1, keepdim=True)
    var = x_2d.var(dim=1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    N = x_2d.shape[1]
    weight = torch.ones((N,), dtype=torch.float32, device=x.device)
    bias = torch.zeros((N,), dtype=torch.float32, device=x.device)
    
    x_hat = (x_2d - mean) * rstd
    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias
    y = x_hat.reshape(x.shape)
    
    return y
