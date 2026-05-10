import torch

@torch.inference_mode()
def torch_bgmv_expand_slice(inputs, lora_b_weights, slice_offset=0, slice_size=64, add_inputs=True):
    """
    inputs: [batch, K]
    lora_b_weights: [num_lora, N, K]
    slice_offset: int (offset into output columns, used for slicing)
    slice_size: int (size of output slice)
    add_inputs: bool (whether to add to existing output)
    Returns: [batch, slice_size]
    """
    lora_indices = torch.tensor([0, 1, -1, 2], dtype=torch.int32, device='cuda')
    batch, K = inputs.shape
    num_lora, N, K2 = lora_b_weights.shape
    assert K == K2
    assert slice_size == N

    output = torch.zeros(batch, slice_size, dtype=inputs.dtype, device=inputs.device)
    for i in range(batch):
        lora_idx = lora_indices[i].item()
        if lora_idx == -1:
            continue
        b = lora_b_weights[lora_idx]  # [N, K]
        out = torch.matmul(b, inputs[i].to(b.dtype))  # [N]
        if add_inputs:
            output[i] = output[i] + out
        else:
            output[i] = out
    return output
