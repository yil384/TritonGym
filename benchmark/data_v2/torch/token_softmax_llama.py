import torch

def torch_token_softmax_llama(Logics):
    """
    Token-wise softmax over variable-length sequences.

    Args:
        Logics: [head_num, total_seq_len] tensor
    Returns:
        Prob_Out: same shape as Logics
    """
    # Split the token axis into `batch_size` contiguous segments derived from
    # the actual width, matching triton_token_softmax_llama.
    batch_size = 2
    total_len = Logics.shape[1]
    base_len, rem = divmod(total_len, batch_size)
    seq_lens = [base_len + (1 if i < rem else 0) for i in range(batch_size)]
    start_locs = [sum(seq_lens[:i]) for i in range(batch_size)]
    B_Start_Loc = torch.tensor(start_locs, dtype=torch.int32, device=Logics.device)
    B_Seqlen = torch.tensor(seq_lens, dtype=torch.int32, device=Logics.device)
    Prob_Out = torch.empty_like(Logics, dtype=Logics.dtype, device=Logics.device)
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
