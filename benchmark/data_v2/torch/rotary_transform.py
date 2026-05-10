# benchmark/data/torch/rotary_transform.py
from typing import Optional, Union
import torch

def torch_rotary_transform(
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
    PyTorch implementation of Triton rotary embedding kernel.
    Supports variable-length sequences, interleaved mode, seqlen_offsets, and conjugate option.

    The Triton kernel computes cos/sin indices as: rm_cs = rm + seqlen_offsets
    where rm = position indices within the sequence. For non-tensor seqlen_offsets,
    the triton wrapper adds seqlen to the int offset before passing to the kernel.
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

    cos, sin = cos.contiguous(), sin.contiguous()

    # Match the triton wrapper's seqlen_offsets handling
    if isinstance(seqlen_offsets, torch.Tensor):
        seqlen_offsets_val = seqlen_offsets
    else:
        # The triton wrapper does: seqlen_offsets += seqlen, then passes as scalar
        seqlen_offsets_val = seqlen_offsets + S

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < D and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    def apply_rotary_to_slice(x_slice, cos_slice, sin_slice):
        """Apply rotary embedding to a [seqlen, rotary_dim] slice."""
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
            # Compute the cos/sin offset for this batch
            if isinstance(seqlen_offsets_val, torch.Tensor):
                offset = seqlen_offsets_val[b].item()
            else:
                offset = seqlen_offsets_val

            # Position indices: rm = 0..S-1, rm_cs = rm + offset
            pos_indices = torch.arange(S, device=x.device) + offset
            # Clamp to valid range and gather cos/sin
            pos_indices = pos_indices.clamp(0, S_ro - 1)
            cos_b = cos[pos_indices]  # [S, rotary_dim_half]
            sin_b = sin[pos_indices]  # [S, rotary_dim_half]

            for h in range(H):
                x_slice = x[b, :S, h, :rotary_dim].float()
                out_slice = apply_rotary_to_slice(x_slice, cos_b, sin_b)
                output[b, :S, h, :rotary_dim] = out_slice.to(x.dtype)
    else:
        for b in range(B):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            seq_len_b = end - start

            if isinstance(seqlen_offsets_val, torch.Tensor):
                offset = seqlen_offsets_val[b].item()
            else:
                offset = seqlen_offsets_val

            pos_indices = torch.arange(seq_len_b, device=x.device) + offset
            pos_indices = pos_indices.clamp(0, S_ro - 1)
            cos_b = cos[pos_indices]
            sin_b = sin[pos_indices]

            for h in range(H):
                x_slice = x[start:end, h, :rotary_dim].float()
                out_slice = apply_rotary_to_slice(x_slice, cos_b, sin_b)
                output[start:end, h, :rotary_dim] = out_slice.to(x.dtype)

    return output
