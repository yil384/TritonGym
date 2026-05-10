import torch

def torch_add_value(x):
    """Add a constant value of 10 to every element of the input tensor.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape as x with 10 added to each element.
    """
    return x + 10

