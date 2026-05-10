import torch

def torch_quantize_copy_kv(K):
    """
    PyTorch equivalent of _fwd_kernel_destindex_copy_quantize_kv Triton kernel.
    K: [seq_len, head_num, head_dim], float16
    DestLoc: [seq_len], int32
    Out: [seq_len, head_num, head_dim], int8 (preallocated)
    Out_scale: [seq_len, head_num, 1], float16 (preallocated)
    """
    seq_len, head_num, head_dim = K.shape
    DestLoc = torch.arange(seq_len, device=K.device, dtype=torch.int32)
    Out = torch.empty_like(K, dtype=torch.int8)
    Out_scale = torch.empty(seq_len, head_num, 1, device=K.device, dtype=K.dtype)

    for cur_index in range(seq_len):
        dest_index = DestLoc[cur_index]
        src_data = K[cur_index, :head_num, :head_dim]  # [head_num, head_dim]
        abs_data = src_data.abs()
        data_scale = (abs_data.max(dim=1, keepdim=True)[0] / 127.).to(torch.float16)
        q_src_data = (src_data / data_scale).round().to(torch.int8)

        Out[dest_index, :head_num, :head_dim] = q_src_data
        Out_scale[dest_index, :head_num, 0] = data_scale.squeeze(-1)

    return Out_scale
