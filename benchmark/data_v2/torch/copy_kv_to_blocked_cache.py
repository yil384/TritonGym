import torch

def torch_copy_kv_to_blocked_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    use_new_kcache_layout: bool = False,
):
    """
    Pure PyTorch version of copy_to_kvcache_seqlen1_kernel.
    Only supports seq_len=1 input (decoding step).
    """
    k = k.squeeze(1) if k.dim() == 4 else k
    v = v.squeeze(1) if v.dim() == 4 else v

    bsz, num_kv_heads, head_dim = k.shape
    max_blocks_per_sequence = 10
    block_size = 16
    k_cache = torch.zeros(max_blocks_per_sequence, num_kv_heads, block_size, head_dim, dtype=k.dtype, device=k.device)
    v_cache = torch.zeros(max_blocks_per_sequence, num_kv_heads, block_size, head_dim, dtype=k.dtype, device=k.device)
    kv_lengths = torch.randint(1, block_size * max_blocks_per_sequence, (bsz,), dtype=torch.int32, device=k.device)
    block_tables = torch.randint(0, max_blocks_per_sequence, (bsz, max_blocks_per_sequence), dtype=torch.int32, device=k.device)
    block_size = k_cache.size(-2)

    for b in range(bsz):
        past_kv_seq_len = kv_lengths[b].item() - 1
        last_block_idx = past_kv_seq_len // block_size
        offset_in_block = past_kv_seq_len % block_size
        block_id = block_tables[b, last_block_idx].item()

        for h in range(num_kv_heads):
            if use_new_kcache_layout:
                # new kcache layout: [num_blocks, num_kv_heads, head_dim//x, block_size, x]
                x = k_cache.size(-1)
                splits = head_dim // x
                for split_idx in range(splits):
                    start = split_idx * x
                    end = (split_idx + 1) * x
                    k_cache[block_id, h, split_idx, offset_in_block, :] = k[b, h, start:end]
                v_cache[block_id, h, offset_in_block, :] = v[b, h, :]
            else:
                # old kcache layout: [num_blocks, num_kv_heads, block_size, head_dim]
                k_cache[block_id, h, offset_in_block, :] = k[b, h, :]
                v_cache[block_id, h, offset_in_block, :] = v[b, h, :]
    return k_cache
