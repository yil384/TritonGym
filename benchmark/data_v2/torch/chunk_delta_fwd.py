import torch

def torch_chunk_delta_fwd(k, w, u, BT=32, initial_state=None, final_state=None):
    """
    PyTorch implementation matching Triton chunk_delta_rule_fwd_kernel_h.
    k: [B, H, T, K] (note: triton wrapper passes k with shape (B,H,T,K))
    w: [B, H, T, K] (d in the triton kernel)
    u: [B, H, T, V] (v in the triton kernel)
    Returns: v_new [B, H, T, V]
    """
    B, H, T, K = k.shape
    V = u.shape[-1]
    NT = (T + BT - 1) // BT
    v_new = torch.zeros_like(u)

    for b in range(B):
        for hh in range(H):
            if initial_state is not None:
                b_h = initial_state[b, hh].clone().float()
            else:
                b_h = torch.zeros(K, V, dtype=torch.float32, device=k.device)

            for nt in range(NT):
                # Triton stores h BEFORE updating in each chunk
                # (we don't return h, only v_new)
                b_h_cumsum = torch.zeros(K, V, dtype=torch.float32, device=k.device)

                t_start = nt * BT
                t_end = min(t_start + BT, T)

                for t_idx in range(t_start, t_end):
                    # k is [B,H,T,K], loaded as (K,T) in triton via transposed strides
                    b_k = k[b, hh, t_idx, :]  # [K]
                    b_d = w[b, hh, t_idx, :]  # [K]
                    b_v = u[b, hh, t_idx, :].float()  # [V]
                    # b_v -= b_d @ b_h
                    b_v = b_v - torch.matmul(b_d.float(), b_h)
                    v_new[b, hh, t_idx, :] = b_v.to(u.dtype)
                    # b_h_cumsum += outer(b_k, b_v)
                    b_h_cumsum += torch.outer(b_k.float(), b_v)

                b_h = b_h + b_h_cumsum

            if final_state is not None:
                final_state[b, hh] = b_h.to(final_state.dtype)

    return v_new
