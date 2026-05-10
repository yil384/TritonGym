import torch

def torch_masked_add_cuda(grad: torch.Tensor, p_data: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    PyTorch implementation of masked add:
        grad += p_data * alpha * (1 - p_mask)
    """
    torch.manual_seed(42)
    p_mask = torch.randint(0, 2, grad.shape, device=grad.device, dtype=torch.int32)
    grad = grad.clone()
    mask = (p_mask == 0)
    grad[mask] += p_data[mask] * alpha
    return grad
