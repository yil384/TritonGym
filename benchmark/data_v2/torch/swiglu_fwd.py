import torch

def torch_swiglu_fwd(xy: torch.Tensor, out: torch.Tensor = None):
    """
    Pure PyTorch version of Swiglu forward.
    xy: [batch, 2*hidden], concatenated X and Y
    """
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)

    if out is None:
        out = torch.empty_like(x)  # Replaces new_empty_like

    out = x * torch.sigmoid(x) * y
    return out.reshape(*batch_shape, x.shape[-1])
