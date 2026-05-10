import torch
from typing import Tuple

def torch_fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch version of fused recurrent HGRN forward pass.
    """
    B, H, T, D = x.shape
    device = x.device
    dtype = x.dtype

    h = torch.zeros(B, H, D, device=device, dtype=torch.float32)
    if initial_state is not None:
        h += initial_state.to(torch.float32)

    o = torch.zeros_like(x, dtype=torch.float32)

    for t in range(T):
        x_t = x[:, :, t, :].to(torch.float32)
        g_t = g[:, :, t, :].to(torch.float32)
        h = g_t * h + x_t
        o[:, :, t, :] = h

    final_state = h if output_final_state else None
    o = o.to(dtype)
    if final_state is not None:
        final_state = final_state.to(dtype)
    return o
