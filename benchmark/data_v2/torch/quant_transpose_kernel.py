import torch

def torch_quant_transpose_kernel(A: torch.Tensor):
    """
    Pure PyTorch version of quantize + transpose.
    Args:
        A: float32 tensor of shape [M, N]
    Returns:
        B: int8 tensor of shape [N, M] (transposed)
        absmax: float32 tensor with global max abs value
    """
    M, N = A.shape
    absmax = A.abs().max()
    absmax_inv = 1.0 / absmax
    # quantize and transpose
    B = torch.clamp((A * absmax_inv * 127.0).round(), -127, 127).to(torch.int8)
    B = B.t().contiguous()  # transpose to [N, M]
    return B
