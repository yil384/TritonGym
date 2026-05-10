import torch


def torch_kcache_copy(
    k: torch.Tensor,
    n: int = 1,
    use_new_kcache_layout: bool = False,
):
    """
    PyTorch implementation of copying keys/values into blocked cache.
    """
    if k.dim() == 4:
        k = k.reshape(-1, k.size(-2), k.size(-1))

    bszn, num_kv_heads, head_dim = k.shape
    max_blocks = 10
    block_size = 16
    k_cache = torch.zeros(max_blocks, num_kv_heads, block_size, head_dim, device=k.device, dtype=k.dtype)
    # Use deterministic values for reproducibility
    torch.manual_seed(42)
    kv_lengths = torch.randint(0, block_size, (bszn,), dtype=torch.int32, device=k.device)
    block_tables = torch.randint(0, max_blocks, (bszn, max_blocks), dtype=torch.int32, device=k.device)
    if n > 1:
        assert bszn % n == 0
        bsz = bszn // n
    else:
        bsz = bszn

    block_size = k_cache.shape[-2] if not use_new_kcache_layout else k_cache.shape[-2]

    for seq_idx in range(bsz):
        for token_offset in range(n):
            token_idx = seq_idx * n + token_offset
            past_len = kv_lengths[seq_idx].item() + token_offset
            block_id = block_tables[seq_idx, past_len // block_size].item()
            pos_in_block = past_len % block_size

            if not use_new_kcache_layout:
                k_cache[block_id, :, pos_in_block, :] = k[token_idx]
            else:
                x = k_cache.size(-1)
                k_reshaped = k[token_idx].view(num_kv_heads, head_dim // x, x)
                k_cache[block_id, :, :, pos_in_block, :] = k_reshaped
    return k_cache
