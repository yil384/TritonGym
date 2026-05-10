import torch

def torch_vector_addition_custom(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of custom vector addition.

    Matches the triton _add_kernel which processes first a.size(0) linear elements.
    """
    c = torch.empty_like(a)
    size = a.size(0)
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    c_flat = c.view(-1)
    c_flat[:size] = a_flat[:size] + b_flat[:size]
    return c
