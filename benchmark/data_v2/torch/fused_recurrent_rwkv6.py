import torch
from typing import Tuple

def torch_fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of fused recurrent RWKV6.

    Args:
        r: (B, H, T, K) query
        k: (B, H, T, K)
        v: (B, H, T, V)
        w: (B, H, T, K)
        u: (H, K)
        scale: scaling factor, default 1/sqrt(K)
        initial_state: optional recurrent state, (B, H, K, V)
        output_final_state: whether to return the final recurrent state

    Returns:
        o: (B, H, T, V)
    """
    B, H, T, K = r.shape
    V = v.shape[-1]
    if scale == -1:
        scale = K ** -0.5

    if initial_state is not None:
        h = initial_state.clone().to(torch.float32)
    else:
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)

    outputs = []
    for t in range(T):
        q_t = r[:, :, t, :].float() * scale        # (B,H,K)
        k_t = k[:, :, t, :].float()                # (B,H,K)
        v_t = v[:, :, t, :].float()                # (B,H,V)
        w_t = torch.exp(w[:, :, t, :].float())     # (B,H,K)

        kv = torch.einsum("bhk,bhv->bhkv", k_t, v_t)   # (B,H,K,V)
        u_broadcast = u.float().unsqueeze(0)  # (1,H,K)
        o_t = torch.einsum("bhkv,bhk->bhv", h + kv * u_broadcast.unsqueeze(-1), q_t)  # (B,H,V)

        outputs.append(o_t)
        h = h * w_t.unsqueeze(-1) + kv

    o = torch.stack(outputs, dim=2)  # (B,H,T,V)

    return o.to(r.dtype)
