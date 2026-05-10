import torch

def torch_quantize_kv_transform(K: torch.Tensor):
    """
    Torch implementation of Triton's quantize_kv_transform.
    Returns (Out, Out_scale) tensors, creating them internally.
    """
    seq_len, head_num, head_dim = K.shape
    DestLoc = torch.randint(0, seq_len, (seq_len,), device=K.device, dtype=torch.int32)

    # Create output tensors
    Out = torch.empty_like(K, dtype=torch.int8)
    Out_scale = torch.empty((seq_len, head_num, 1), dtype=K.dtype, device=K.device)

    # Convert to float32 for safe max computation
    K_float = K.float()
    abs_data = K_float.abs()
    scale = abs_data.max(dim=2, keepdim=True)[0] / 127.0
    scale[scale == 0] = 1e-8  # prevent division by zero

    # Quantize - Triton does truncation (.to(tl.int8)), not rounding
    q_data = (K_float / scale).to(torch.int8)

    # Scatter according to DestLoc
    for i in range(seq_len):
        dest_idx = DestLoc[i].item()
        Out[dest_idx, :, :] = q_data[i, :, :]
        Out_scale[dest_idx, :, 0] = scale[i, :, 0]

    return Out_scale
