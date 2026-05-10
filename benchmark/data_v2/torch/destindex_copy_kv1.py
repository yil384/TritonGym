import torch

def torch_destindex_copy_kv1(K: torch.Tensor):
    """
    Copy each row from K to Out according to DestLoc indices.
    K: [seq_len, head_num, head_dim]
    DestLoc: [seq_len] (destination indices)
    Out: [seq_len, head_num, head_dim] (pre-allocated)
    """
    seq_len, head_num, head_dim = K.shape
    DestLoc = torch.arange(seq_len, dtype=torch.int32, device=K.device)
    Out = torch.zeros_like(K)
    # Iterate over rows and copy
    for i in range(seq_len):
        dest_idx = DestLoc[i].item()
        Out[dest_idx, :head_num, :head_dim] = K[i, :head_num, :head_dim]
    return Out
