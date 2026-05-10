import torch

def torch_destindex_copy_kv2(K: torch.Tensor):
    """
    Copy each row from K to Out according to DestLoc indices.
    K: [seq_len, head_num, head_dim]
    DestLoc: [seq_len] (destination indices)
    Out: [seq_len, head_num, head_dim] (pre-allocated)
    """
    seq_len, head_num, head_dim = K.shape
    DestLoc = torch.arange(seq_len, dtype=torch.int32, device=K.device)
    Out = torch.zeros_like(K)
    # Convert indices to int64
    dest_indices = DestLoc.to(torch.int64).view(-1, 1, 1).expand(-1, head_num, head_dim)
    Out.scatter_(0, dest_indices, K)
    return Out
