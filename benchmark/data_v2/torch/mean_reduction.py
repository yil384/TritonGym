import torch

def torch_mean_reduction(x: torch.Tensor, dim = [1, 2], keepdim=False, *, dtype=None):
    """
    Torch implementation of mean reduction along specified dimensions.
    Mirrors functionality of triton_mean_reduction.
    """
    if dtype is None:
        dtype = x.dtype

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % x.ndim for d in dim]  # support negative dims

    # Compute mean along the given dims
    out = x.to(dtype).mean(dim=dim, keepdim=keepdim)
    return out
