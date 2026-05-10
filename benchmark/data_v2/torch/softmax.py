import torch

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch implementation of the Triton softmax kernel.
    Args:
        x: Input tensor of shape (n_rows, n_cols)
    Returns:
        Softmax of x with the same shape
    """
    # Compute max for numerical stability
    row_max, _ = x.max(dim=1, keepdim=True)
    numerator = torch.exp(x - row_max)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator
