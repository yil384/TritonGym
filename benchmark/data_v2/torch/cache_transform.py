import torch

def torch_cache_transform(cos_cache, sin_cache, is_prompts=False):
    """Transform rotary embedding caches by gathering position-specific cos/sin values.

    In prompt mode, flattens sequences and maps each token to its within-sequence
    position. In decode mode, selects the last-position embedding for each sequence.

    Args:
        cos_cache: Cosine cache of shape (max_length, hidden_dim).
        sin_cache: Sine cache of shape (max_length, hidden_dim).
        is_prompts: If True, use prompt (prefill) logic; otherwise use decode logic.

    Returns:
        In prompt mode: tuple of (cos_output, sin_output) each of shape (total_length, hidden_dim).
        In decode mode: cos_output of shape (num_seqs, hidden_dim).
    """
    max_length = cos_cache.shape[0]
    num_seqs = 8  # assume num_seqs=8 for prompts
    lengths = torch.randint(1, max_length, (num_seqs,), dtype=torch.int32, device=cos_cache.device)
    if is_prompts:
        cumsum_lengths = torch.cumsum(lengths, dim=0)
        total_length = cumsum_lengths[-1].item()
        hidden_dim = cos_cache.shape[1]
        cos_output = torch.zeros(total_length, hidden_dim, device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.zeros(total_length, hidden_dim, device=sin_cache.device, dtype=sin_cache.dtype)
        for idx in range(total_length):
            # Replicate triton logic: ori_seq_idx = idx - max(cumsum_lengths[cumsum_lengths <= idx])
            valid = cumsum_lengths[cumsum_lengths <= idx]
            if valid.numel() == 0:
                ori_seq_idx = idx
            else:
                ori_seq_idx = idx - valid.max().item()
            cos_output[idx] = cos_cache[ori_seq_idx]
            sin_output[idx] = sin_cache[ori_seq_idx]
        return cos_output, sin_output
    else:
        # decoding case: use lengths - 1 as position indices
        num_seqs = lengths.numel()
        hidden_dim = cos_cache.shape[1]
        cos_output = torch.zeros((num_seqs, hidden_dim), device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.zeros((num_seqs, hidden_dim), device=sin_cache.device, dtype=sin_cache.dtype)
        for i in range(num_seqs):
            cos_output[i] = cos_cache[lengths[i]-1]
            sin_output[i] = sin_cache[lengths[i]-1]
        return cos_output
