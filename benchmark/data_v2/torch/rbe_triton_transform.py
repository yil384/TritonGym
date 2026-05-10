import torch

def torch_rbe_triton_transform(x: torch.Tensor, pos: int = 0) -> torch.Tensor:
    """
    Pure PyTorch implementation of rbe_triton kernel (Rotary Block Embedding).
    Args:
        x: input tensor of shape (batch, M, K)
        pos: starting token position
    Returns:
        out: tensor of same shape as x
    """
    batch, M, K = x.shape
    DIM = 128  # hardcoded in triton kernel
    theta = 10000.0  # hardcoded in triton wrapper

    device = x.device
    dtype = x.dtype

    out = x.clone()

    # Even indices: 0, 2, 4, ...
    even_idx = torch.arange(0, K, 2, device=device)  # shape: (K//2,)

    # Frequency computation matching triton's get_freq_multi_tokens
    # freqs = offs_cn % DIM (offs_cn are even indices)
    freq_idx = (even_idx % DIM).float() / DIM  # shape: (K//2,)
    freq = theta ** freq_idx  # shape: (K//2,)

    # Token positions
    token_pos = (pos + torch.arange(M, device=device)).float()  # shape: (M,)

    # freqs = token_pos / freq -> shape: (M, K//2)
    freqs = token_pos[:, None] / freq[None, :]

    cos_val = torch.cos(freqs)  # (M, K//2)
    sin_val = torch.sin(freqs)  # (M, K//2)

    # real = x[..., 0::2], imag = x[..., 1::2]
    real = x[:, :, 0::2].float()  # (batch, M, K//2)
    imag = x[:, :, 1::2].float()  # (batch, M, K//2)

    # Rotation
    out_real = real * cos_val.unsqueeze(0) - imag * sin_val.unsqueeze(0)
    out_imag = real * sin_val.unsqueeze(0) + imag * cos_val.unsqueeze(0)

    out[:, :, 0::2] = out_real.to(dtype)
    out[:, :, 1::2] = out_imag.to(dtype)

    return out
