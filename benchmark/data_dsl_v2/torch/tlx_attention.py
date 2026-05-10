import torch

def torch_tlx_attention(q, k, v, sm_scale=1.3):
    """Compute scaled dot-product attention and cast output to float16.

    Args:
        q: Query tensor of shape (B, num_heads, seq_len, head_dim).
        k: Key tensor of shape (B, num_heads, seq_len, head_dim).
        v: Value tensor of shape (B, num_heads, seq_len, head_dim).
        sm_scale: Scale factor applied to QK^T scores before softmax.

    Returns:
        Attention output tensor of shape (B, num_heads, seq_len, head_dim) in float16.
    """
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p, dim=-1)
    out = torch.matmul(p, v).half()
    return out

if __name__ == "__main__":
    input_shapes = [
        ((4, 32, 2048, 128,),(4, 32, 2048, 128,),(4, 32, 2048, 128,),),
        ((4, 32, 4096, 128,),(4, 32, 4096, 128,),(4, 32, 4096, 128,),),
    ]
    