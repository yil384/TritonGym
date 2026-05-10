import torch

def torch_softmax_flaggems(x: torch.Tensor, dim: int = -1, dtype=None) -> torch.Tensor:
    """
    Pure PyTorch version of Triton Softmax for both forward and backward.
    Supports arbitrary dimensions, float16/float32/float64.
    """
    if dtype is None:
        dtype = x.dtype
    out = torch.softmax(x.float(), dim=dim)
    return out.to(dtype)
