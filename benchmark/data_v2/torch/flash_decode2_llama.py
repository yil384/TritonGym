import torch

def torch_flash_decode2_llama(mid_out: torch.Tensor, mid_out_logexpsum: torch.Tensor,
                              O: torch.Tensor, block_seq: int=8):
    """
    PyTorch version of Flash Decode Stage 2.

    Args:
        mid_out: [batch, head, seq_block_num, head_dim]
        mid_out_logexpsum: [batch, head, seq_block_num]
        B_Seqlen: [batch] effective sequence length per batch.
        block_seq: BLOCK_SEQ size.
        O: [batch, head, head_dim] output tensor.
    """
    batch_size, head_num, seq_block_num, head_dim = mid_out.shape
    device = mid_out.device
    dtype = mid_out.dtype
    B_Seqlen = torch.randint(0, seq_block_num * block_seq, (batch_size,), device=device, dtype=torch.int32)

    for b in range(batch_size):
        seq_len = B_Seqlen[b].item()
        block_n_size = 0 if seq_len <= 0 else (seq_len + block_seq - 1) // block_seq

        for h in range(head_num):
            acc = torch.zeros(head_dim, device=device, dtype=dtype)
            max_logic = torch.tensor(-float("inf"), device=device, dtype=dtype)
            sum_exp = torch.tensor(0.0, device=device, dtype=dtype)

            for blk in range(block_n_size):
                tv = mid_out[b, h, blk]  # [head_dim]
                tlogic = mid_out_logexpsum[b, h, blk]  # scalar

                new_max_logic = torch.max(tlogic, max_logic)
                old_scale = torch.exp(max_logic - new_max_logic)
                acc = acc * old_scale
                exp_logic = torch.exp(tlogic - new_max_logic)
                acc = acc + exp_logic * tv
                sum_exp = sum_exp * old_scale + exp_logic
                max_logic = new_max_logic

            O[b, h] = acc / sum_exp
    return O
