import torch
import torch.nn as nn

def torch_fast_rms_layernorm(X: torch.Tensor, gemma: bool = False):
    """
    Pure PyTorch implementation of Fast RMS LayerNorm (from Triton version)
    Supports optional GEMMA variant: output = normed * (W + 1)
    """
    layernorm = torch.nn.LayerNorm(X.size(-1), elementwise_affine=True).to(X.device)
    W = layernorm.weight
    eps = getattr(layernorm, "variance_epsilon", getattr(layernorm, "eps", 1e-5))

    # RMS normalization: norm = X / sqrt(mean(X^2) + eps)
    mean_square = (X * X).mean(dim=-1, keepdim=True)
    inv_var = 1.0 / torch.sqrt(mean_square + eps)
    normed = X * inv_var

    if gemma:
        out = normed * (W + 1.0)
    else:
        out = normed * W
    return out
