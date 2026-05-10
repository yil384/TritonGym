import torch
from typing import Optional, Tuple

def torch_rotary_emb_nopad(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Safe Torch implementation of rotary embedding without in-place operations.

    Args:
        q: [total_tokens, q_head_num, head_dim]
        k: [total_tokens, kv_head_num, head_dim]
        cos, sin: [max_position_len, head_dim]
        k_cache: optional key cache, [num_blocks, kv_head_num, block_size, head_dim]
        block_tables: optional block table
        kv_lengths: optional kv lengths
    Returns:
        q_rot, k_rot, k_cache_rot (if k_cache is not None)
    """
    q_total_tokens, q_head_num, head_dim = q.shape
    k_head_num = k.shape[1]

    # compute kv group number
    kv_group_num = q_head_num // k_head_num
    assert q_head_num % k_head_num == 0

    # prepare output tensors
    q_out = q.clone()
    k_out = k.clone()
    if k_cache is not None:
        k_cache_out = k_cache.clone()
    else:
        k_cache_out = None

    # rotary embedding
    for token_idx in range(q_total_tokens):
        cos_row = cos[token_idx, :head_dim//2].to(q.dtype)
        sin_row = sin[token_idx, :head_dim//2].to(q.dtype)

        for qh in range(q_head_num):
            q0 = q_out[token_idx, qh, :head_dim//2]
            q1 = q_out[token_idx, qh, head_dim//2:]
            q_out[token_idx, qh, :head_dim//2] = q0 * cos_row - q1 * sin_row
            q_out[token_idx, qh, head_dim//2:] = q0 * sin_row + q1 * cos_row

            # Handle corresponding k
            if qh % kv_group_num == 0:
                kh = qh // kv_group_num
                k0 = k_out[token_idx, kh, :head_dim//2]
                k1 = k_out[token_idx, kh, head_dim//2:]
                k_out[token_idx, kh, :head_dim//2] = k0 * cos_row - k1 * sin_row
                k_out[token_idx, kh, head_dim//2:] = k0 * sin_row + k1 * cos_row

    return q_out
