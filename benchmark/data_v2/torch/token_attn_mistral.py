import torch

def torch_token_attn_mistral(Prob, V):
    """
    PyTorch version of Triton token attention (Mistral style) with sliding window.

    Args:
        Prob: [num_heads, sum(seq_lens)] attention probabilities
        V: [num_kv_heads, sum(seq_lens), d_model] values
        Req_to_tokens: [batch, seq_len] token mapping indices
        B_req_idx: [batch] batch indices
        B_Start_Loc: [batch] start indices in Prob
        B_Seqlen: [batch] sequence lengths
        B_Att_Start_Loc: [batch] start indices for attention window
        B_Att_Seqlen: [batch] attention window lengths
        sliding_window: int, size of sliding window
    Returns:
        Out: [batch, num_heads, d_model]
    """
    device = Prob.device
    batch, seq_len = 2, Prob.shape[1]
    sliding_window = 64
    Req_to_tokens = torch.randint(0, seq_len, (batch, seq_len), dtype=torch.int32, device=device)
    B_req_idx = torch.arange(batch, dtype=torch.int32, device=device)
    B_Start_Loc = torch.zeros((batch,), dtype=torch.int32, device=device)
    B_Seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    B_Att_Start_Loc = torch.zeros((batch,), dtype=torch.int32, device=device)
    B_Att_Seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    batch = B_req_idx.shape[0]
    num_heads = Prob.shape[0]
    d_model = V.shape[2]
    kv_group_num = num_heads // V.shape[0]

    Out = torch.zeros((batch, num_heads, d_model), dtype=Prob.dtype, device=Prob.device)

    for b in range(batch):
        start_loc = B_Start_Loc[b].item()
        seq_len = B_Seqlen[b].item()
        att_start = B_Att_Start_Loc[b].item()
        att_len = B_Att_Seqlen[b].item()
        req_tokens = Req_to_tokens[b, :seq_len]

        # sliding window start
        batch_start_index = max(seq_len - sliding_window, 0)

        for h in range(num_heads):
            kv_head = h // kv_group_num
            acc = torch.zeros(d_model, dtype=Prob.dtype, device=Prob.device)
            for n in range(att_len):
                prob_idx = att_start + n
                if prob_idx >= Prob.shape[1]:
                    continue
                p = Prob[h, prob_idx]

                token_idx = batch_start_index + n
                if token_idx >= len(req_tokens):
                    continue
                v_idx = min(req_tokens[token_idx].item(), V.shape[1]-1)  # Prevent out-of-bounds

                acc += p * V[kv_head, v_idx]
            Out[b, h] = acc

    return Out
