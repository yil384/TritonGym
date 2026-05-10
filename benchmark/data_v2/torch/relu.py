import torch

def torch_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch replacement for triton_relu.
    Args:
        x: input tensor
    Returns:
        Tensor after ReLU
    """
    return torch.clamp(x, min=0)
