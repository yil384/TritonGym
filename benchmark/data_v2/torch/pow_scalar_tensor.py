import torch

def torch_pow_scalar_tensor(in0: torch.Tensor, val0: float = 2.0) -> torch.Tensor:
    """
    Pure PyTorch implementation: val0 ^ in0 (scalar raised to tensor power), elementwise.
    """
    return torch.pow(val0, in0.float()).to(in0.dtype)
