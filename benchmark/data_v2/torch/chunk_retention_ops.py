import torch
from typing import Tuple

def torch_chunk_retention_ops(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    scale: float = None,
    checkpoint_level: int = 1,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of chunk retention operator.
    q, k: [B, H, T, K]
    v: [B, H, T, V]
    initial_state: [B, H, K, V] or None
    output_final_state: bool (ignored in torch impl, triton returns final_state separately)
    scale: float
    checkpoint_level: int (ignored in torch impl)

    The triton kernel uses per-head exponential decay:
        b_b = log2(1 - 2^(-5 - h))  for head index h
        decay_per_step = 2^(b_b)

    Returns: o [B, H, T, V]
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    device, dtype = q.device, q.dtype

    if scale is None:
        scale = K ** -0.5

    o = torch.zeros((B, H, T, V), device=device, dtype=torch.float32)

    # Compute per-head decay factors (matching triton kernel)
    head_indices = torch.arange(H, device=device, dtype=torch.float64)
    b_b = torch.log2(1.0 - torch.pow(2.0, -5.0 - head_indices))  # [H]
    decay = torch.pow(2.0, b_b).float()  # [H], decay per timestep

    # H_cache: [B, H, K, V]
    if initial_state is not None:
        H_cache = initial_state.float().clone()
    else:
        H_cache = torch.zeros((B, H, K, V), device=device, dtype=torch.float32)

    for t_idx in range(T):
        k_t = k[:, :, t_idx, :].float()  # [B, H, K]
        v_t = v[:, :, t_idx, :].float()  # [B, H, V]
        q_t = q[:, :, t_idx, :].float()  # [B, H, K]

        # In the triton kernel, the decay for key-value outer product uses
        # d_i = 2^((BT - i - 1) * b_b) within each chunk, and the
        # cumulative state gets multiplied by d_b = 2^(BT * b_b).
        # For a step-by-step reference, the recurrence is:
        # H_t = decay * H_t-1 + k_t^T v_t * decay_factor_for_position
        # But the exact chunk-level computation involves intra-chunk attention too.
        #
        # Simplified step-by-step recurrence (matching the overall semantics):
        d = decay.unsqueeze(0)  # [1, H]
        H_cache = d.unsqueeze(-1).unsqueeze(-1) * H_cache + \
                  k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # [B, H, K, V]

        # o_t = q_t @ H_cache * scale
        o[:, :, t_idx, :] = torch.einsum('bhk,bhkv->bhv', q_t, H_cache) * scale

    return o.to(dtype)
