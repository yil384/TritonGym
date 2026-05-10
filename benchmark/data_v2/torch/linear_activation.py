import torch
import math
from typing import Optional

sqrt2pi = math.sqrt(2.0 / math.pi)
sqrt2 = math.sqrt(2.0)


def torch_tanh(x: torch.Tensor) -> torch.Tensor:
    return x.tanh()


def torch_relu(x: torch.Tensor) -> torch.Tensor:
    return x.relu()


def torch_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / sqrt2))


def torch_fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch_tanh(sqrt2pi * (x + 0.044715 * x**3)))


def torch_linear_activation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "relu",
    act_inputs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure PyTorch replacement of Triton LinearLayer + activation kernel.
    :param x: Input tensor (M, K)
    :param weight: Weight tensor (N, K)
    :param bias: Optional bias (N,)
    :param activation: Optional activation: 'relu', 'tanh', 'gelu', 'fast_gelu'
    :param act_inputs: Optional tensor to store pre-activation outputs
    :return: Output tensor (M, N)
    """
    x_ = x if x.ndim == 2 else x.flatten(0, 1)
    M, K = x_.shape
    N, K_ = weight.shape
    assert K == K_, f"Incompatible shapes: x={x_.shape}, weight={weight.shape}"
    act_inputs = torch.empty(M, N, device=x.device, dtype=x.dtype)

    out = x_ @ weight.T
    if bias is not None:
        out = out + bias

    if act_inputs is not None:
        act_inputs.copy_(out)

    if activation == "relu":
        out = torch_relu(out)
    elif activation == "tanh":
        out = torch_tanh(out)
    elif activation == "gelu":
        out = torch_gelu(out)
    elif activation == "fast_gelu":
        out = torch_fast_gelu(out)

    if x.ndim != 2:
        out = out.reshape(*x.shape[:-1], N)

    return out
