import torch
import math
from typing import Optional


def torch_silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def torch_llama_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch replacement of Triton ff_llama kernel:
    out = silu(x @ w1.T) * (x @ w3.T) with RMS normalization
    :param x: (batch, seq_len, dim)
    :param w1: (dim, dim)
    :param w3: (dim, dim)
    :param rms_w: (dim,)
    :return: (batch, seq_len, dim)
    """
    batch, seq_len, dim = x.shape
    M = batch * seq_len

    x_flat = x.reshape(M, dim)

    # Compute RMS normalization (eps=1e-6 hardcoded to match triton kernel)
    eps = 1e-6
    a_sum = torch.sum(x_flat.float() ** 2, dim=1)  # (M,)
    a_mean = a_sum / dim + eps
    a_norm = torch.rsqrt(a_mean)  # (M,)

    # Apply rms_w element-wise to input, then project through w1 and w3
    # Triton kernel: a = x * rms_w, acc = a @ w1_t, then acc *= a_norm
    x_rms = x_flat * rms_w  # (M, dim)
    acc1 = x_rms @ w1.T  # (M, N)
    acc2 = x_rms @ w3.T  # (M, N)

    # Apply RMS normalization factor
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]

    # Apply SiLU to acc1, then multiply with acc2
    out = torch_silu(acc1) * acc2

    return out.view(batch, seq_len, -1)
