import torch
from typing import Optional, Tuple

def torch_chunk_linear_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: Optional[float] = None, initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False, normalize: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Torch implementation of chunked linear attention matching the Triton version.

    The algorithm:
    1. Compute running state h[t] = sum_{i<t*BT} k_i^T @ v_i (accumulated per chunk)
    2. For each chunk: o = (q @ h) + causal_mask(q @ k^T) @ v, scaled

    q, k: [B, H, T, K]
    v:    [B, H, T, V]
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    BT = 64
    NT = (T + BT - 1) // BT

    if initial_state is None:
        initial_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    else:
        initial_state = initial_state.to(torch.float32)

    o = torch.empty_like(v)

    for b in range(B):
        for hh in range(H):
            # Running state: [K, V]
            b_h = initial_state[b, hh].clone()

            for nt in range(NT):
                t_start = nt * BT
                t_end = min(t_start + BT, T)
                chunk_len = t_end - t_start

                q_chunk = q[b, hh, t_start:t_end, :].float()  # [chunk_len, K]
                k_chunk = k[b, hh, t_start:t_end, :].float()  # [chunk_len, K]
                v_chunk = v[b, hh, t_start:t_end, :].float()  # [chunk_len, V]

                # Inter-chunk: o += q @ h (state from previous chunks)
                b_o = torch.matmul(q_chunk, b_h)  # [chunk_len, V]

                # Intra-chunk: causal attention within the chunk
                b_s = torch.matmul(q_chunk, k_chunk.t())  # [chunk_len, chunk_len]
                # Causal mask: only attend to positions <= current
                idx = torch.arange(chunk_len, device=q.device)
                mask = idx[:, None] >= idx[None, :]
                b_s = torch.where(mask, b_s, torch.zeros_like(b_s))

                b_o = (b_o + torch.matmul(b_s, v_chunk)) * scale
                o[b, hh, t_start:t_end, :] = b_o.to(v.dtype)

                # Update state: h += k^T @ v
                b_h = b_h + torch.matmul(k_chunk.t(), v_chunk)

    return o
