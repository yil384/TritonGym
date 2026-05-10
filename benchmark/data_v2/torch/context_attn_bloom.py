# benchmark/data/torch/context_attn_bloom.py

import torch
import torch.nn.functional as F
from typing import Optional


def torch_context_attn_bloom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Torch implementation of context attention (Bloom-like), compatible with Triton version.
    Args:
        q: [tokens, n_heads, d_head]
        k: [tokens_total, n_kv_heads, d_head]
        v: [tokens_total, n_kv_heads, d_head]
        b_req_idx: [B]
        b_start_loc: [B]
        b_seq_len: [B]
        b_prompt_cache_len: [B]
        max_input_len: int
        req_to_token_indexs: [B, total_tokens]
    Returns:
        out: [tokens, n_heads, d_head]
    """
    device = q.device
    tokens, _, _ = q.shape
    B = k.shape[0] // tokens
    T = tokens // B
    extra_kv = k.shape[0] - B * T
    # meta info
    b_start_loc = torch.arange(0, B * T, T, device=device, dtype=torch.int32)
    b_seq_len = torch.full((B,), T, device=device, dtype=torch.int32)
    b_req_idx = torch.arange(B, device=device, dtype=torch.int32)
    b_prompt_cache_len = torch.zeros(B, device=device, dtype=torch.int32)
    req_to_token_indexs = torch.stack(
        [torch.arange(0, T + extra_kv, device=device, dtype=torch.int32) for _ in range(B)], dim=0
    )
    max_input_len = T
    B = b_seq_len.shape[0]
    _, H, D = q.shape
    _, H_k, _ = k.shape
    kv_group_num = H // H_k

    out = torch.zeros_like(q)

    for b in range(B):
        start = b_start_loc[b].item()
        seqlen = b_seq_len[b].item()
        prompt_cache_len = b_prompt_cache_len[b].item()
        req_idx = b_req_idx[b].item()

        # Index range for the current batch
        token_indices = req_to_token_indexs[b, : seqlen + prompt_cache_len]

        q_b = q[start : start + seqlen]   # [T, H, D]
        k_b = k[token_indices]            # [T', H_k, D]
        v_b = v[token_indices]            # [T', H_k, D]

        # Computescaled dot-product attention
        scale = 1.0 / (D ** 0.5)
        # reshape KV
        k_b = k_b.repeat_interleave(kv_group_num, dim=1)   # expand heads
        v_b = v_b.repeat_interleave(kv_group_num, dim=1)

        attn_weights = torch.einsum("thd,shd->ths", q_b, k_b) * scale
        # Mask: keep only the causal part
        causal_mask = torch.tril(torch.ones_like(attn_weights, dtype=torch.bool))
        attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        o_b = torch.einsum("ths,shd->thd", attn_probs, v_b)

        out[start : start + seqlen] = o_b

    return out