import torch

def torch_gluon_memcpy_1d(input, XBLOCK=1024, layout=None, num_warps=4):
    """Copy a 1D tensor into a new tensor of the same shape and dtype.

    Args:
        input: Input 1D tensor of shape (N,).
        XBLOCK: Unused Triton block size parameter.
        layout: Unused Triton layout parameter.
        num_warps: Unused Triton warp count parameter.

    Returns:
        A new tensor containing a copy of the input data.
    """
    output = torch.empty_like(input)
    output.copy_(input)
    return output