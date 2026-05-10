import torch

def torch_dropout(x, x_keep, p: float=0.5):
    """
    PyTorch version of Dropout logic, simulating the Triton kernel.
    """
    assert x.is_contiguous()
    output = torch.where(x_keep == 1, x / (1 - p), torch.zeros_like(x))
    return output
