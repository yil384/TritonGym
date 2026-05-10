import torch

def torch_rope_transform(q, k, cos, sin):
    """
    PyTorch version of RoPE forward transform.
    q, k: (batch, n_head, seq_len, head_dim)
    cos, sin: (seq_len, head_dim//2)
    """
    batch, n_q_head, seq_len, head_dim = q.shape
    half_dim = head_dim // 2

    cos_ = cos.view(1, 1, seq_len, half_dim)
    sin_ = sin.view(1, 1, seq_len, half_dim)

    # Compute in fp32 for precision
    q_f = q.float()
    cos_f = cos_.float()
    sin_f = sin_.float()

    q1, q2 = q_f[..., :half_dim], q_f[..., half_dim:]
    q_new = torch.cat([q1 * cos_f - q2 * sin_f, q2 * cos_f + q1 * sin_f], dim=-1)

    return q_new.to(q.dtype)
