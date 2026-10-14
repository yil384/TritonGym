import torch

def torch_token_attn_reduceV(Prob, V):
    """
    Reduce V values weighted by Prob using token-wise attention.
    Prob: [num_heads, total_seq_len]
    V: [num_kv_heads, total_seq_len, d_model]
    Req_to_tokens: [batch, max_seq_len]
    """
    device = Prob.device
    batch, seq_len = 2, Prob.shape[1] // 2
    # Same fixed-seed token mapping as the Triton wrapper; indices address V's token dim.
    gen = torch.Generator().manual_seed(0)
    Req_to_tokens = torch.randint(0, V.shape[1], (batch, seq_len), dtype=torch.int32, generator=gen).to(device)
    B_req_idx = torch.arange(batch, dtype=torch.int32, device=device)
    B_Start_Loc = torch.zeros(batch, dtype=torch.int32, device=device)
    B_Seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    batch = B_req_idx.shape[0]
    num_heads = Prob.shape[0]
    d_model = V.shape[2]
    num_kv_heads = V.shape[0]
    kv_group_num = num_heads // num_kv_heads

    Out = torch.zeros((batch, num_heads, d_model), dtype=Prob.dtype, device=Prob.device)

    for b in range(batch):
        start_loc = B_Start_Loc[b].item()
        seqlen = B_Seqlen[b].item()
        req_tokens = Req_to_tokens[b, :seqlen]

        for h in range(num_heads):
            kv_head = h // kv_group_num
            acc = torch.zeros(d_model, dtype=torch.float32, device=Prob.device)
            for n in range(seqlen):
                p = Prob[h, start_loc + n]
                v_idx = req_tokens[n].item()
                acc += p.float() * V[kv_head, v_idx].float()
            Out[b, h] = acc.to(Out.dtype)
    return Out
