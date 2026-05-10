import torch

def torch_per_block_int8(q, k, BLKQ=128, BLKK=64):
    """
    Pure PyTorch implementation of per-block int8 quantization for query and key.
    Returns: q_int8, q_scale, k_int8, k_scale
    """
    q_shape_orig = q.shape
    k_shape_orig = k.shape
    q = q.reshape(-1, q.shape[-2], q.shape[-1])
    k = k.reshape(-1, k.shape[-2], k.shape[-1])

    B, L, C = q.shape

    # Allocate int8 outputs and scales
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)

    q_scale = torch.empty((B, (L + BLKQ - 1) // BLKQ, 1), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((B, (L + BLKK - 1) // BLKK, 1), device=k.device, dtype=torch.float32)

    # Per-block processing for query
    for b in range(B):
        for blk in range((L + BLKQ - 1) // BLKQ):
            start = blk * BLKQ
            end = min(start + BLKQ, L)
            x = q[b, start:end, :].float()
            x = x * (C ** -0.5 * 1.44269504)
            scale = x.abs().amax() / 127.
            x_div = x / scale
            x_div = x_div + 0.5 * torch.sign(x_div)
            x_int8 = x_div.to(torch.int8)
            q_int8[b, start:end, :] = x_int8
            q_scale[b, blk, 0] = scale

    # Per-block processing for key
    for b in range(B):
        for blk in range((L + BLKK - 1) // BLKK):
            start = blk * BLKK
            end = min(start + BLKK, L)
            x = k[b, start:end, :].float()
            scale = x.abs().amax() / 127.
            x_div = x / scale
            x_div = x_div + 0.5 * torch.sign(x_div)
            x_int8 = x_div.to(torch.int8)
            k_int8[b, start:end, :] = x_int8
            k_scale[b, blk, 0] = scale

    # Restore original shape
    q_int8 = q_int8.reshape(q_shape_orig)
    k_int8 = k_int8.reshape(k_shape_orig)

    return q_scale
