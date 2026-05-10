import torch

def torch_kldiv_compute(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence elementwise: x * log(x / y)
    Uses abs + clamp to handle negative random inputs gracefully.
    """
    x_pos = torch.abs(x).clamp(min=1e-7)
    y_pos = torch.abs(y).clamp(min=1e-7)
    return x_pos * torch.log(x_pos / y_pos)
