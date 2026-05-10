import torch

def torch_gluon_add(a, b, XBLOCK=32, YBLOCK=64, num_buffers=2):
    """Compute element-wise addition of two tensors.

    Args:
        a: First input tensor of arbitrary shape.
        b: Second input tensor, broadcastable to the shape of a.
        XBLOCK: Unused Triton block size parameter.
        YBLOCK: Unused Triton block size parameter.
        num_buffers: Unused Triton buffer count parameter.

    Returns:
        Element-wise sum tensor of the same shape as a.
    """
    c = a + b
    return c