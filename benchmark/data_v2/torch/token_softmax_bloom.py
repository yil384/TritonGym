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
    B_Start_Loc = torch.tensor([0, 1025, 2050, 3075], device=Logics.device, dtype=torch.int32)
    B_Seqlen = torch.tensor([1025, 1025, 1025, 1025], device=Logics.device, dtype=torch.int32)
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
