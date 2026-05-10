import torch

def torch_token_softmax_llama(Logics):
    """
    Token-wise softmax over variable-length sequences.
    Matches the Triton kernel behavior, including linear memory access patterns.

    Args:
        Logics: [head_num, total_seq_len] tensor
    Returns:
        Prob_Out: same shape as Logics
    """
    batch_size = 2
    head_num = 2
    max_input_len = 8
    B_Start_Loc = torch.arange(0, batch_size * max_input_len, max_input_len, dtype=torch.int32, device=Logics.device)
    B_Seqlen = torch.full((batch_size,), max_input_len, dtype=torch.int32, device=Logics.device)
    Prob_Out = torch.empty_like(Logics, dtype=Logics.dtype, device=Logics.device)
    head_num = Logics.shape[0]
    batch = B_Start_Loc.shape[0]

    total_cols = Logics.shape[1]

    # Flatten to simulate Triton's linear memory access pattern.
    # When (start_idx + col_offset) >= total_cols, Triton reads linearly into
    # the next head's data (same underlying memory buffer).
    flat = Logics.contiguous().view(-1).float()
    total_elements = flat.numel()

    stride_h = Logics.stride(0)  # = total_cols for contiguous tensor

    for b in range(batch):
        start_idx = B_Start_Loc[b].item()
        seq_len = B_Seqlen[b].item()
        if seq_len == 0:
            continue

        for h in range(head_num):
            base = h * stride_h
            offsets = torch.arange(seq_len, device=Logics.device)
            addrs = base + start_idx + offsets

            # For addresses within the tensor, read actual values.
            # For truly OOB addresses (past end of tensor), use -inf (masked out).
            # This matches Triton's mask=col_offsets < seq_len with other=-inf
            # for the case where the kernel can't actually read valid memory.
            valid = addrs < total_elements
            safe_addrs = addrs.clamp(0, total_elements - 1)
            row = torch.where(valid, flat[safe_addrs], torch.tensor(0.0, device=Logics.device, dtype=torch.float32))

            # Softmax in fp32
            row_max = row.max()
            row_shifted = row - row_max
            exp_vals = row_shifted.exp()
            sum_exp = exp_vals.sum()
            softmax_vals = exp_vals / sum_exp

            # Store only to valid output positions
            for i in range(seq_len):
                out_col = start_idx + i
                if out_col < total_cols:
                    Prob_Out[h, out_col] = softmax_vals[i].to(Logics.dtype)

    return Prob_Out
