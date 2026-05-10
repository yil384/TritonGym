import torch
from typing import Literal

def torch_fill_kv_cache(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    k_caches: torch.Tensor,
    v_caches: torch.Tensor,
    k_scales_zeros: torch.Tensor = None,
    v_scales_zeros: torch.Tensor = None,
    quant_policy: Literal[0, 4, 8] = 0
):
    """
    PyTorch version of Triton fill_kv_cache kernel.
    Only supports BLOCK = last dim of k_caches.
    """
    batch_size = k_states.size(0)
    block_size = k_caches.size(1)
    num_heads = k_caches.size(2)
    head_dim = k_caches.size(3)
    head_dim_v = v_states.size(3)

    max_q_seq_length = k_states.size(1)
    q_start_loc = torch.zeros(batch_size, dtype=torch.int32, device=k_states.device)
    q_seq_length = torch.full((batch_size,), max_q_seq_length, dtype=torch.int32, device=k_states.device)
    kv_seq_length = torch.full((batch_size,), max_q_seq_length, dtype=torch.int32, device=k_states.device)
    block_offsets = torch.zeros((batch_size, max_q_seq_length // block_size + 1), dtype=torch.int32, device=k_states.device)


    # Flatten k_states/v_states to match triton's linear pointer arithmetic
    # Triton uses: ks_ptr = KStates + cur_startloc * stride_kss (no batch offset)
    k_states_flat = k_states.reshape(-1, num_heads, head_dim)
    v_states_flat = v_states.reshape(-1, num_heads, head_dim_v)

    for b in range(batch_size):
        q_start = q_start_loc[b].item()
        q_len = q_seq_length[b].item()
        kv_len = kv_seq_length[b].item()
        history_len = kv_len - q_len
        block0_first_tokenloc = history_len % block_size
        max_block_id = (q_len + block_size - 1) // block_size

        for block_id in range(max_block_id):
            state_token_offset = max(block_id * block_size - block0_first_tokenloc, 0)
            kv_block_id = max((history_len + 1 + block_size - 1)//block_size - 1 + block_id, 0)
            kv_block_id = min(kv_block_id, block_offsets.size(1) - 1)
            block_off = block_offsets[b, kv_block_id].item()
            cur_startloc = q_start + state_token_offset

            c_first_tokenloc = block0_first_tokenloc if block_id == 0 else 0
            c_last_tokenloc = min(block_size, q_len + block0_first_tokenloc - block_id * block_size)

            for bidx in range(c_first_tokenloc, c_last_tokenloc):
                sidx = bidx - c_first_tokenloc
                k_slice = k_states_flat[cur_startloc + sidx, :, :]
                if quant_policy == 4:
                    k_slice = ((k_slice / k_slice.max()) * 15).round().to(torch.uint8)
                elif quant_policy == 8:
                    k_slice = ((k_slice / k_slice.max()) * 255).round().to(torch.uint8)
                k_caches[block_off, bidx, :, :] = k_slice

                v_slice = v_states_flat[cur_startloc + sidx, :, :]
                if quant_policy == 4:
                    v_slice = ((v_slice / v_slice.max()) * 15).round().to(torch.uint8)
                elif quant_policy == 8:
                    v_slice = ((v_slice / v_slice.max()) * 255).round().to(torch.uint8)
                v_caches[block_off, bidx, :, :] = v_slice
    return k_caches