import torch

def torch_l2_norm_bwd(x: torch.Tensor, dy: torch.Tensor, eps: float = 1e-5):
    """
    PyTorch version of Triton L2 backward kernel.
    Computes gradient of L2 normalization along the last dimension.
    """
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    
    # Compute variance and reciprocal of sqrt(var + eps)
    var = torch.sum(x * x, dim=1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    
    # Compute dx
    dx = dy * rstd - (torch.sum(dy * x, dim=1, keepdim=True) / (var + eps)) * rstd * x
    dx = dx.reshape(orig_shape)
    return dx
