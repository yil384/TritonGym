import torch

def torch_l2_norm_fwd(x: torch.Tensor, eps: float = 1e-6):
    """
    PyTorch version of Triton L2 forward kernel.
    Computes L2 normalization along the last dimension.
    """
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    
    # Compute in fp32 for numerical stability
    x_f = x.float()
    var = torch.sum(x_f * x_f, dim=1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    
    y = x_f * rstd
    y = y.to(x.dtype)
    y = y.reshape(orig_shape)
    return y
