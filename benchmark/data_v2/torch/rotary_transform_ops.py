# benchmark/data/torch/rotary_transform_ops.py
import torch
from typing import Union, Optional

def torch_rotary_transform_ops(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """
    PyTorch version of Triton rotary embedding with variable-length support.

    The Triton kernel computes position indices as: rm_cs = rm + seqlen_offsets
    where rm are position indices within the sequence.
    For scalar seqlen_offsets, the triton wrapper checks seqlen_offsets + seqlen <= seqlen_ro.
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        B, S, H, D = x.shape
    else:
        total_S, H, D = x.shape
        B = cu_seqlens.shape[0] - 1
        S = max_seqlen
    S_ro, rotary_dim_half = cos.shape
    rotary_dim = rotary_dim_half * 2

    assert rotary_dim <= D, "rotary_dim must be <= head_dim"
    assert S_ro >= S, "rotary embedding sequence length too small"
    assert x.dtype == cos.dtype == sin.dtype, "x, cos, sin must have same dtype"

    cos, sin = cos.contiguous(), sin.contiguous()
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < D and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    def apply_rotary_to_slice(x_slice, cos_slice, sin_slice):
        """Apply rotary embedding. x_slice: [..., rotary_dim], cos/sin: [..., rotary_dim_half]"""
        s = -sin_slice if conjugate else sin_slice
        if interleaved:
            x0 = x_slice[..., 0::2].clone()
            x1 = x_slice[..., 1::2].clone()
            out = torch.empty_like(x_slice)
            out[..., 0::2] = x0 * cos_slice - x1 * s
            out[..., 1::2] = x0 * s + x1 * cos_slice
            return out
        else:
            half = rotary_dim // 2
            x0 = x_slice[..., :half].clone()
            x1 = x_slice[..., half:rotary_dim].clone()
            out = torch.empty_like(x_slice)
            out[..., :half] = x0 * cos_slice - x1 * s
            out[..., half:rotary_dim] = x0 * s + x1 * cos_slice
            return out

    if not is_varlen:
        for b in range(B):
            # Compute cos/sin position indices with seqlen_offsets
            if isinstance(seqlen_offsets, torch.Tensor):
                offset = seqlen_offsets[b].item()
            else:
                offset = seqlen_offsets

            pos_indices = torch.arange(S, device=x.device) + offset
            pos_indices = pos_indices.clamp(0, S_ro - 1)
            cos_b = cos[pos_indices]  # [S, rotary_dim_half]
            sin_b = sin[pos_indices]

            for h in range(H):
                x_slice = x[b, :S, h, :rotary_dim].float()
                out_slice = apply_rotary_to_slice(x_slice, cos_b, sin_b)
                output[b, :S, h, :rotary_dim] = out_slice.to(x.dtype)
    else:
        for b in range(B):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            seq_len_b = end - start

            if isinstance(seqlen_offsets, torch.Tensor):
                offset = seqlen_offsets[b].item()
            else:
                offset = seqlen_offsets

            # For varlen, position indices start from 0 within each sequence
            pos_indices = torch.arange(seq_len_b, device=x.device) + offset
            pos_indices = pos_indices.clamp(0, S_ro - 1)
            cos_b = cos[pos_indices]
            sin_b = sin[pos_indices]

            for h in range(H):
                x_slice = x[start:end, h, :rotary_dim].float()
                out_slice = apply_rotary_to_slice(x_slice, cos_b, sin_b)
                output[start:end, h, :rotary_dim] = out_slice.to(x.dtype)

    return output
