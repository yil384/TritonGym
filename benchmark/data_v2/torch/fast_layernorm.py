import torch
import torch.nn.functional as F

def torch_fast_layernorm(X: torch.Tensor):
    """
    Pure PyTorch implementation of Fast LayerNorm (from Triton version)
    Supports elementwise affine.
    """
    layernorm = torch.nn.LayerNorm(X.size(-1), elementwise_affine=True).to(X.device)
    assert layernorm.elementwise_affine is True, "Only elementwise_affine=True supported"

    W = layernorm.weight
    b = layernorm.bias
    eps = getattr(layernorm, "variance_epsilon", getattr(layernorm, "eps", 1e-5))

    mean = X.mean(dim=-1, keepdim=True)
    var = X.var(dim=-1, unbiased=False, keepdim=True)
    inv_var = 1.0 / torch.sqrt(var + eps)
    X_norm = (X - mean) * inv_var
    Y = X_norm * W + b
    return Y
