import torch

def torch_context_attn_llama(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """
    Simplified PyTorch version of LLaMA context attention forward.
    q: [batch_seq, H, D]
    k,v: [batch_kv_seq, H, D]
    out: [batch_seq, H, D]
    b_start_loc, b_seq_len, b_prompt_cache_len, b_req_idx: [batch]
    req_to_token_indexs: [num_requests, seq_len]
    """
    device = q.device
    out = torch.zeros_like(q)
    batch_seq, H, D = q.shape
    batch_kv = k.shape[0]
    sm_scale = 1.0 / (D ** 0.5) * 1.4426950408889634  # scale for exp2
    b_start_loc = torch.zeros((1,), dtype=torch.int32, device=device)
    b_seq_len = torch.tensor([batch_seq], dtype=torch.int32, device=device)
    b_prompt_cache_len = torch.zeros((1,), dtype=torch.int32, device=device)
    b_req_idx = torch.zeros((1,), dtype=torch.int32, device=device)
    req_to_token_indexs = torch.arange(batch_seq, device=device).unsqueeze(0).expand(1, -1)
    batch = b_seq_len.shape[0]
    H = q.shape[1]
    D = q.shape[2]

    sm_scale = 1.0 / (D ** 0.5) * 1.4426950408889634  # scale for exp2

    for b in range(batch):
        start = b_start_loc[b].item()
        seq_len = b_seq_len[b].item()
        prompt_cache_len = b_prompt_cache_len[b].item()
        req_idx = b_req_idx[b].item()

        # Slice q for current batch
        q_batch = q[start:start+seq_len, :, :]  # [seq_len, H, D]

        # Compute attention per head
        for h in range(H):
            kv_head = h  # assuming kv_group_num = 1
            k_slice = k[:, h, :]  # [seq_kv, D]
            v_slice = v[:, h, :]  # [seq_kv, D]

            # Map requests to token indices
            idxs = req_to_token_indexs[req_idx, :seq_len]  # [seq_len]
            k_selected = k_slice[idxs, :]
            v_selected = v_slice[idxs, :]

            # Compute scaled dot-product attention
            qk = torch.matmul(q_batch[:, h, :], k_selected.T) * sm_scale

            # Apply causal mask
            mask = torch.arange(seq_len, device=q.device)[:, None] >= torch.arange(seq_len, device=q.device)[None, :]
            qk = torch.where(mask, qk, torch.tensor(-1e8, device=q.device, dtype=q.dtype))

            # Softmax in numerically stable way using max subtraction
            m_i = qk.max(dim=1, keepdim=True).values
            qk_exp = 2 ** (qk - m_i)
            denom = qk_exp.sum(dim=1, keepdim=True)
            p = qk_exp / denom

            out[start:start+seq_len, h, :] = torch.matmul(p, v_selected)
    return out