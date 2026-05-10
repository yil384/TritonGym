import torch

def torch_gluon_copy_scalar(input):
    """Copy a scalar (or tensor) into a new tensor of the same shape and dtype.

    Args:
        input: Input tensor of arbitrary shape.

    Returns:
        A new tensor containing a copy of the input data.
    """
    output = torch.empty_like(input)
    output.copy_(input)
    return output