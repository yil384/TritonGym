import math

import torch
import torch.nn.functional as F

DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

def torch_tlx_simplicial_attention_pipelined(
    q,
    k1,
    k2,
    v1,
    v2,
    w1=32,
    w2=128,
):
    """Reference impl for simplicial attention (pipelined variant).
    w1, w2: Int. Positive values for block-local windowed attention.

    Given [batch, sequence, local_heads, embedding] inputs,
    forms a attention probability cube [batch, sequence, sequence, sequence]
    where each logits[b, n, i, j, k] = sum_h(q[b, i, n, h] * k1[b, j, n, h] * k2[b, k, n, h])
    for i - w1 < j <= i and i - w2 < k <= i

    Then for each i, gets softmax across j, k
    And uses probabilities to lookup against values:
    output[b, i, n, h] = sum_jk(prob[b, n, i, j, k] * v1[b, j, n, h] * v2[b, k, n, h])
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    b, s, k, d = q.shape
    sm_scale = 1.0 / math.sqrt(d)

    # Default triple product.
    logits = torch.einsum("btkh,bskh,brkh->bktsr", q, k1, k2)

    # If block local.
    if w1 > 0 and w2 > 0:
        # Create [q, kv1, kv2] shaped mask.
        q_idx = torch.arange(s)[:, None, None].to(
            device=DEVICE
        )
        kv1_idx = torch.arange(s)[None, :, None].to(
            device=DEVICE
        )
        kv2_idx = torch.arange(s)[None, None, :].to(
            device=DEVICE
        )

        kv1_mask = ((q_idx - w1) < kv1_idx) & (kv1_idx <= q_idx)
        kv2_mask = ((q_idx - w2) < kv2_idx) & (kv2_idx <= q_idx)
        local_mask = kv1_mask & kv2_mask
        logits = torch.where(local_mask[None, None, :], logits, -float("inf"))

    # Per q token softmax. [b, k, t, t, t] -> [b, k, t, t*t]
    logits = logits * sm_scale
    shape = logits.shape
    logits_reshaped = logits.view(*shape[:-2], -1)
    attn_prob = F.softmax(logits_reshaped, dim=-1).type_as(q)
    attn_prob = attn_prob.view(*shape)
    output = torch.einsum("bktsr,bskh,brkh->btkh", attn_prob, v1, v2)

    return output

if __name__ == "__main__":
    input_shapes = [
        ((2, 32, 128, 128),(2, 32, 128, 128),(2, 32, 128, 128),(2, 32, 128, 128),(2, 32, 128, 128),),
        ((4, 32, 128, 128),(4, 32, 128, 128),(4, 32, 128, 128),(4, 32, 128, 128),(4, 32, 128, 128),),
    ]
