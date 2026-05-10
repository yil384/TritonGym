import torch

class TorchLogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=-1, dtype=None):
        if dtype is None:
            dtype = x.dtype
        x = x.to(dtype)
        out = torch.nn.functional.log_softmax(x, dim=dim)
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        dim = ctx.dim
        grad_output = grad_output.to(out.dtype)
        softmax = out.exp()
        grad_input = grad_output - softmax * grad_output.sum(dim=dim, keepdim=True)
        return grad_input, None, None


def torch_log_softmax(x: torch.Tensor, dim=-1, dtype=None) -> torch.Tensor:
    """Compute log-softmax with a custom autograd function supporting forward and backward.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute log-softmax.
        dtype: Optional dtype to cast x to before computation.

    Returns:
        Log-softmax output tensor of the same shape as x.
    """
    return TorchLogSoftmax.apply(x, dim, dtype)
