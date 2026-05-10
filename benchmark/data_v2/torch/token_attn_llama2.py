import torch

def torch_token_attn_llama2(Q, K):
    """
    PyTorch implementation of Triton token attention (LLAMA2 style).

    Args:
        Q: [batch, head, max_input_len, d_model]
        K: [batch, head, max_input_len, d_model]
        B_Loc: [batch, max_input_len] location indices for K
        B_Start_Loc: [batch] start index for attention output
        B_Seqlen: [batch] sequence lengths
        max_input_len: int, maximum sequence length
    Returns:
        Att_Out: [batch, head, max_input_len]
    """
    batch, head, seq_len, d_model = Q.shape
    max_input_len = K.shape[2]
    device = Q.device
    Att_Out = torch.zeros((batch, head, seq_len), dtype=Q.dtype, device=Q.device)
    sm_scale = 1.0 / (d_model ** 0.5)
    B_Loc = torch.randint(0, max_input_len, (batch, max_input_len), dtype=torch.int32, device=device)
    B_Start_Loc = torch.zeros((batch,), dtype=torch.int32, device=device)
    B_Seqlen = torch.randint(1, max_input_len+1, (batch,), dtype=torch.int32, device=device)

    for b in range(batch):
        cur_seq_len = B_Seqlen[b].item()
        start_loc = B_Start_Loc[b].item()
        batch_start_index = max_input_len - cur_seq_len
        for h in range(head):
            for n in range(cur_seq_len):
                q = Q[b, h, n]  # [d_model]
                k_idx = B_Loc[b, batch_start_index + n].item()
                k_idx = min(k_idx, K.shape[2]-1)  # Prevent out-of-bounds
                k = K[b, h, k_idx]  # [d_model]
                att = torch.dot(q, k) * sm_scale
                Att_Out[b, h, start_loc + n] = att

    return Att_Out
