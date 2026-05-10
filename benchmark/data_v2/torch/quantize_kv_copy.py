import torch

def torch_quantize_kv_copy(K: torch.Tensor):
    """
    PyTorch implementation of KV quantize.
    K: [seq_len, head_num, head_dim], float16
    Returns quantized int8 tensor as float16 (for comparison).
    """
    quant_group_dim = 8
    seq_len, head_num, head_dim = K.shape
    group_size = head_dim // quant_group_dim

    K_view = K.view(seq_len, head_num, group_size, quant_group_dim)

    # Compute per-group scale
    abs_data = K_view.abs()
    data_scale = abs_data.max(dim=3).values / 127.0  # [seq_len, head_num, group_size]
    data_scale = data_scale.clamp(min=1e-10)

    # Quantize
    q_data = (K_view / data_scale.unsqueeze(-1)).clamp(-127, 127).round()

    # Dequantize back to float for comparison
    deq_data = q_data * data_scale.unsqueeze(-1)

    return deq_data.view(seq_len, head_num, head_dim).to(K.dtype)
