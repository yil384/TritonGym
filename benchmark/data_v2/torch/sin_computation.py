import torch

def torch_sin_computation(x: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch version of triton_sin_computation.
    Computes elementwise sin of `x` and stores result in `out`.
    """
    # Ensure same shape
    out = torch.empty_like(x)
    assert x.shape == out.shape
    out.copy_(torch.sin(x))
    return out
