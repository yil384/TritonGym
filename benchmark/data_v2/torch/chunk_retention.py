import torch
import math
from typing import Optional, Tuple

def torch_chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Torch reference implementation of chunk_retention operator.
    Matches the Triton chunked retention with exponential decay per head.

    q: [B, H, T, K]
    k: [B, H, T, K]
    v: [B, H, T, V]
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5
    BT = 64
    NT = (T + BT - 1) // BT

    o = torch.empty_like(v)

    for b in range(B):
        for hh in range(H):
            # Per-head decay rate (matches triton: log2(1 - 2^(-5 - h)))
            b_b = math.log2(1 - 2 ** (-5 - hh * 1.0))

            # Running state [K, V]
            if initial_state is not None:
                b_h = initial_state[b, hh].clone().float()
            else:
                b_h = torch.zeros(K, V, dtype=torch.float32, device=q.device)

            for nt in range(NT):
                t_start = nt * BT
                t_end = min(t_start + BT, T)
                chunk_len = t_end - t_start

                q_chunk = q[b, hh, t_start:t_end, :].float()  # [chunk_len, K]
                k_chunk = k[b, hh, t_start:t_end, :].float()  # [chunk_len, K]
                v_chunk = v[b, hh, t_start:t_end, :].float()  # [chunk_len, V]

                o_i = torch.arange(chunk_len, device=q.device, dtype=torch.float32)

                # Decay for inter-chunk: d_i = 2^((i+1)*b_b)
                d_i = (2.0 ** ((o_i + 1) * b_b))  # [chunk_len]

                # Inter-chunk: o = (q * d_i) @ h
                b_o = torch.matmul((q_chunk * d_i[:, None]), b_h)  # [chunk_len, V]

                # Intra-chunk: causal attention with decay
                b_s = torch.matmul(q_chunk, k_chunk.t())  # [chunk_len, chunk_len]
                # Causal mask with decay: d_s[i,j] = 2^((i-j)*b_b) if i>=j else 0
                mask = o_i[:, None] >= o_i[None, :]
                d_s = torch.where(mask, 2.0 ** ((o_i[:, None] - o_i[None, :]) * b_b), torch.zeros_like(b_s))
                b_s = b_s * d_s

                b_o = (b_o + torch.matmul(b_s.to(v_chunk.dtype), v_chunk)) * scale
                o[b, hh, t_start:t_end, :] = b_o.to(v.dtype)

                # Update state: h = d_b * h + k^T @ (v * d_decay)
                d_b = 2.0 ** (chunk_len * b_b)
                d_decay = 2.0 ** ((chunk_len - o_i - 1) * b_b)  # [chunk_len]
                b_h = d_b * b_h + torch.matmul(k_chunk.t(), (v_chunk * d_decay[:, None]))

    return o
