import torch
from collections import namedtuple

def torch_max(inp: torch.Tensor):
    """
    Core Torch implementation of 1D max reduction (equivalent to triton max).
    """
    # Return a named tuple with values and indices
    Max_out = namedtuple("max", ["values", "indices"])
    max_val, max_idx = torch.max(inp, dim=0)
    return Max_out(values=max_val, indices=max_idx)


def torch_max_reduction(inp: torch.Tensor, dim=1, keepdim=False):
    """
    Core Torch implementation of max along a given dimension (equivalent to triton_max_reduction)
    """
    if dim is None:
        # flatten and reduce all elements
        max_val = inp.max()
        max_idx = inp.flatten().argmax()
        Max_out = namedtuple("max", ["values", "indices"])
        return Max_out(values=max_val, indices=max_idx)
    
    dim = dim % inp.ndim
    max_val, max_idx = torch.max(inp, dim=dim, keepdim=keepdim)
    Max_out = namedtuple("max", ["values", "indices"])
    return Max_out(values=max_val, indices=max_idx)
