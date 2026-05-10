import torch
from typing import Tuple

def torch_fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of fused recurrent retention.
    Args:
        q: (B, H, T, Dqk)
        k: (B, H, T, Dqk)
        v: (B, H, T, Dv)
        initial_state: optional, (B, H, Dqk, Dv)
        output_final_state: whether to return the final recurrent state

    Returns:
        o: (B, H, T, Dv)
        final_state: (B, H, Dqk, Dv) if output_final_state else None
    """
    B, H, T, Dqk = q.shape
    Dv = v.shape[-1]
    scale = Dqk ** -0.5

    # Initialize recurrent state h
    if initial_state is not None:
        h = initial_state.clone().to(torch.float32)
    else:
        h = torch.zeros(B, H, Dqk, Dv, dtype=torch.float32, device=q.device)

    # Precompute decay coefficients
    b_h = torch.arange(H, device=q.device, dtype=torch.float32)
    decay = 1 - torch.exp2(-5 - b_h)
    decay = decay.view(1, H, 1, 1)  # (1,H,1,1)

    outputs = []
    for t in range(T):
        q_t = q[:, :, t, :].float() * scale  # (B,H,Dqk)
        k_t = k[:, :, t, :].float()          # (B,H,Dqk)
        v_t = v[:, :, t, :].float()          # (B,H,Dv)

        # Update recurrent state
        h = decay * h + torch.einsum("bhd,bhe->bhde", k_t, v_t)  # (B,H,Dqk,Dv)

        # Compute output
        o_t = torch.einsum("bhd,bhde->bhe", q_t, h)  # (B,H,Dv)
        outputs.append(o_t)

    o = torch.stack(outputs, dim=2).to(q.dtype)  # (B,H,T,Dv)

    final_state = h if output_final_state else None
    return o
