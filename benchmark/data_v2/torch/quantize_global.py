import torch

def torch_quantize_global(x: torch.Tensor):
    """
    PyTorch equivalent of _quantize_global Triton kernel.
    x: [*], float32 or float16
    Returns:
        output: int8 tensor of same shape as x
        absmax: float tensor of shape [1]
    """
    absmax = x.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    output = (x * absmax_inv * 127.0).round().to(torch.int8)
    return output
