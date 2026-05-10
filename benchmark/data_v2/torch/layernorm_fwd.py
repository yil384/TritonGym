import torch

def torch_layernorm_fwd(X, W, eps=1e-5):
    """
    Torch implementation of the Triton 3D LayerNorm forward kernel.
    
    Args:
        X: input tensor of shape [Batch, Heads, Features]
        W: weight tensor of shape [Heads, Features]
        eps: small float for numerical stability
    
    Returns:
        y: normalized output, same shape as X
    """
    # Compute in fp32 for numerical stability
    X_f = X.float()
    mean = X_f.mean(dim=-1, keepdim=True)
    var = ((X_f - mean) ** 2).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    
    y = (X_f - mean) * rstd * W.float()
    return y.to(X.dtype)
