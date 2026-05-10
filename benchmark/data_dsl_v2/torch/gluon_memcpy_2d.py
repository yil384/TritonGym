import torch

def torch_gluon_memcpy_2d(input, XBLOCK=1024, YBLOCK=16, layout=None, num_warps=1):
    """Copy a 2D tensor into a new tensor of the same shape and dtype.

    Args:
        input: Input 2D tensor of shape (M, N).
        XBLOCK: Unused Triton block size parameter.
        YBLOCK: Unused Triton block size parameter.
        layout: Unused Triton layout parameter.
        num_warps: Unused Triton warp count parameter.

    Returns:
        A new tensor containing a copy of the input data.
    """
    output = torch.empty_like(input)
    output.copy_(input)
    return output
