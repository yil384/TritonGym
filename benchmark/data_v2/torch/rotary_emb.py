# benchmark/data/torch/rotary_emb.py
import torch
from typing import Optional, Tuple

def torch_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rotary_factor: float = 1.0
) -> torch.Tensor:
    """
    Pure PyTorch version of rotary embedding matching the Triton kernel.

    The Triton kernel uses interleaved (even/odd) indexing for cos/sin:
      - dim_range0 = arange(0, BLOCK_DMODEL//2) * 2      (even indices: 0,2,4,...)
      - dim_range1 = arange(0, BLOCK_DMODEL//2) * 2 + 1   (odd indices: 1,3,5,...)
      - cos0 = Cos[t, even], sin0 = Sin[t, even]
      - cos1 = Cos[t, odd],  sin1 = Sin[t, odd]
      - out_even = q_even * cos0 - q_odd * sin0
      - out_odd  = q_even * sin1 + q_odd * cos1

    Args:
        q: [total_len, head_num_q, head_dim]
        k: [total_len, head_num_k, head_dim]
        cos: [total_len, head_dim]
        sin: [total_len, head_dim]
        partial_rotary_factor: fraction of head_dim to apply rotary to

    Returns:
        q_out: rotated query tensor
    """
    total_len, head_num_q, head_dim = q.shape
    head_num_k = k.shape[1]
    rotary_dim = int(head_dim * partial_rotary_factor)

    q_out = q.clone()
    k_out = k.clone()

    half_dim = rotary_dim // 2

    # Extract interleaved cos/sin values
    # cos has shape [total_len, head_dim], even indices = cos0, odd indices = cos1
    cos0 = cos[:, 0::2][:, :half_dim]  # [total_len, half_dim]
    sin0 = sin[:, 0::2][:, :half_dim]
    cos1 = cos[:, 1::2][:, :half_dim]
    sin1 = sin[:, 1::2][:, :half_dim]

    # Apply to q: q has shape [total_len, head_num_q, head_dim]
    # even dim indices
    q_even = q[:, :, 0::2][:, :, :half_dim]  # [total_len, head_num_q, half_dim]
    q_odd = q[:, :, 1::2][:, :, :half_dim]

    # Broadcast cos/sin: [total_len, 1, half_dim]
    cos0_ = cos0.unsqueeze(1)
    sin0_ = sin0.unsqueeze(1)
    cos1_ = cos1.unsqueeze(1)
    sin1_ = sin1.unsqueeze(1)

    q_out[:, :, 0::2] = q_even * cos0_ - q_odd * sin0_
    q_out[:, :, 1::2] = q_even * sin1_ + q_odd * cos1_

    # Apply to k
    k_even = k[:, :, 0::2][:, :, :half_dim]
    k_odd = k[:, :, 1::2][:, :, :half_dim]

    k_out[:, :, 0::2] = k_even * cos0_ - k_odd * sin0_
    k_out[:, :, 1::2] = k_even * sin1_ + k_odd * cos1_

    return q_out
