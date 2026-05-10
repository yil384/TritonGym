import torch

def torch_vector_addition(x, y):
    """Compute element-wise addition of two tensors.

    Args:
        x: First input tensor of shape (N,).
        y: Second input tensor of shape (N,).

    Returns:
        Element-wise sum tensor of shape (N,).
    """
    return x + y

