import torch

def torch_rope_backward_transform(dq: torch.Tensor, dk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    PyTorch version equivalent to triton_rope_backward_transform.

    Args:
        dq: Tensor of shape (batch_size, n_q_head, seq_len, head_dim).
        dk: Tensor of shape (batch_size, n_kv_head, seq_len, head_dim).
        cos, sin: Tensor of shape (seq_len, head_dim // 2).

    Returns:
        dq_out, dk_out: Tensor with same shape as dq/dk.
    """
    # Transpose to (batch, seq_len, n_head, head_dim)
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    def rope_backward(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        cos_ = cos.unsqueeze(1)  # (seq_len, 1, head_dim//2)
        sin_ = sin.unsqueeze(1)
        # Inverse rotary transform
        x1_new = x1 * cos_ + x2 * sin_
        x2_new = x2 * cos_ - x1 * sin_
        return torch.cat([x1_new, x2_new], dim=-1)

    dq_out = rope_backward(dq)
    dk_out = rope_backward(dk)

    # Transpose back to original shape (batch, n_head, seq_len, head_dim)
    return dq_out.transpose(1, 2), dk_out.transpose(1, 2)
