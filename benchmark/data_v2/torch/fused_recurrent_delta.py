import torch
from typing import Tuple

def torch_fused_recurrent_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute fused recurrent attention with the delta rule update.

    Maintains a hidden state h of shape (B, H, K, V) that is updated at each
    timestep using the delta rule: h += k_t * (beta * (v_t - h^T k_t)).

    Args:
        q: Query tensor of shape (B, H, T, K).
        k: Key tensor of shape (B, H, T, K).
        v: Value tensor of shape (B, H, T, V).
        beta: Optional gating tensor; broadcastable to (B, H, T, V).
        scale: Optional query scale factor; defaults to K^{-0.5}.
        initial_state: Optional initial hidden state of shape (B, H, K, V).
        output_final_state: If True, return the final hidden state (unused in return).

    Returns:
        Output tensor of shape (B, H, T, V).
    """
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if beta is None:
        beta = torch.ones(B, H, T, V, device=q.device, dtype=q.dtype)
    headwise_beta = beta.ndim == v.ndim

    # Initialize hidden state
    h = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    if initial_state is not None:
        h += initial_state.to(torch.float32)

    o = torch.zeros(B, H, T, V, device=q.device, dtype=torch.float32)

    for t in range(T):
        q_t = q[:, :, t, :].to(torch.float32) * scale  # [B,H,K]
        k_t = k[:, :, t, :].to(torch.float32)          # [B,H,K]
        v_t = v[:, :, t, :].to(torch.float32)          # [B,H,V]
        beta_t = beta[:, :, t, :].to(torch.float32) if headwise_beta else beta[:, :, t:t+1]

        # Delta rule update
        # Compute _v_minus = sum over K: h * k_t, with proper broadcasting
        _v_minus = (h * k_t.unsqueeze(-1)).sum(dim=2)  # [B,H,V]
        v_t = v_t - _v_minus
        v_t = v_t * beta_t

        # Update hidden state
        h = h + k_t.unsqueeze(-1) * v_t.unsqueeze(2)  # [B,H,K,V]

        # Compute output: sum over K: h * q_t
        o[:, :, t, :] = (h * q_t.unsqueeze(-1)).sum(dim=2)  # [B,H,V]

    final_state = h if output_final_state else None
    o = o.to(v.dtype)
    return o
