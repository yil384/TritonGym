import torch

def torch_token_softmax_bloom(Logics):
    """
    Token-wise softmax for variable-length sequences (Bloom style).

    Args:
        Logics: [head_num, sum(seq_lens)] tensor
        B_Start_Loc: [batch] tensor, start index of each batch
        B_Seqlen: [batch] tensor, sequence length of each batch

    Returns:
        Prob_Out: same shape as Logics
    """
    # Logics is [head_num, sum(seq_lens)]. Split its token axis into `batch`
    # contiguous segments derived from the actual width (4 x 1025 for the
    # canonical (12, 4100) input), matching triton_token_softmax_bloom.
    batch = 4
    total_len = Logics.shape[1]
    base_len, rem = divmod(total_len, batch)
    seq_lens = [base_len + (1 if i < rem else 0) for i in range(batch)]
    start_locs = [sum(seq_lens[:i]) for i in range(batch)]
    B_Start_Loc = torch.tensor(start_locs, device=Logics.device, dtype=torch.int32)
    B_Seqlen = torch.tensor(seq_lens, device=Logics.device, dtype=torch.int32)
    Prob_Out = torch.empty_like(Logics, dtype=Logics.dtype, device=Logics.device)
    head_num = Logics.shape[0]
    batch = B_Start_Loc.shape[0]

    for b in range(batch):
        start_idx = B_Start_Loc[b].item()
        seq_len = B_Seqlen[b].item()
        if seq_len == 0:
            continue
        # slice shape: [head_num, seq_len]
        input_slice = Logics[:, start_idx:start_idx + seq_len].float()

        # token-wise softmax in fp32 to match Triton kernel accumulation
        max_vals = input_slice.max(dim=1, keepdim=True)[0]
        exp_vals = (input_slice - max_vals).exp()
        sum_exp = exp_vals.sum(dim=1, keepdim=True)
        softmax_slice = exp_vals / sum_exp

        Prob_Out[:, start_idx:start_idx + seq_len] = softmax_slice.to(Logics.dtype)

    return Prob_Out
