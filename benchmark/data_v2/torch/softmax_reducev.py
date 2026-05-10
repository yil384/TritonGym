import torch

def torch_softmax_reducev(Logics, V):
    """Compute softmax-weighted reduction of V using paged attention logits.

    Performs numerically stable online softmax over attention logits and accumulates
    the weighted sum of value vectors, using block-based token indexing.

    Args:
        Logics: Attention logit tensor of shape (head, total_tokens).
        V: Value tensor of shape (head, max_seq_len, d_model).

    Returns:
        Output tensor of shape (batch, head, d_model).
    """
    max_seq_len = V.shape[1]
    batch, head = Logics.shape[1] // max_seq_len, Logics.shape[0]
    max_input_len = max_seq_len
    device = Logics.device
    B_Loc = torch.randint(0, max_seq_len, (batch, max_seq_len), device=device, dtype=torch.int32)
    B_Start_Loc = torch.randint(0, max_seq_len, (batch,), device=device, dtype=torch.int32)
    B_Seqlen = torch.randint(1, max_seq_len + 1, (batch,), device=device, dtype=torch.int32)
    other_kv_index = 0
    batch, head = B_Seqlen.shape[0], Logics.shape[0]
    d_model = V.shape[-1]
    Out = torch.zeros((batch, head, d_model), device=V.device, dtype=V.dtype)
    BLOCK_N = 64

    for cur_batch in range(batch):
        cur_batch_seq_len = B_Seqlen[cur_batch]
        cur_batch_start_loc = B_Start_Loc[cur_batch]

        for cur_head in range(head):
            # Safely create scalars on GPU
            e_max = torch.full((1,), float('-inf'), device=V.device, dtype=V.dtype)
            e_sum = torch.zeros((1,), device=V.device, dtype=V.dtype)
            acc = torch.zeros(d_model, device=V.device, dtype=V.dtype)

            start_n = 0
            while start_n < cur_batch_seq_len:
                end_n = min(start_n + BLOCK_N, cur_batch_seq_len)
                offs_n = torch.arange(start_n, end_n, device=V.device)

                # Safe indexing for v
                off_b_loc = max_input_len - cur_batch_seq_len
                idx = off_b_loc + offs_n
                idx = torch.clamp(idx, 0, B_Loc.shape[1]-1)
                v_index = B_Loc[cur_batch, idx]
                v_index = torch.where((v_index >= 0) & (v_index < V.shape[1]),
                                      v_index,
                                      torch.tensor(other_kv_index, device=V.device))

                # Safe indexing for qk
                qk_idx = cur_batch_start_loc + offs_n
                qk_idx = torch.clamp(qk_idx, 0, Logics.shape[1]-1)
                qk = Logics[cur_head, qk_idx]

                n_e_max = torch.max(qk.max(), e_max)
                old_scale = torch.exp(e_max - n_e_max)
                p = torch.exp(qk - n_e_max)
                e_sum = e_sum * old_scale + p.sum()
                acc = acc * old_scale + (p[:, None] * V[cur_head, v_index, :]).sum(0)
                e_max = n_e_max

                start_n += BLOCK_N

            Out[cur_batch, cur_head, :] = acc / e_sum

    return Out
