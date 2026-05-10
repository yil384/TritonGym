import torch

def torch_sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
) -> None:
    """
    PyTorch implementation of sgmv_expand_slice.
    inputs: [token_nums, slice_size]
    lora_b_weights: [lora_num, slice_size, rank] or [lora_num, 1, slice_size, rank]
    output_tensor: [token_nums, slice_size]
    b_seq_start_loc, seq_len_tensor, lora_indices_tensor: [batch]
    slice_offset: start column offset in output
    add_inputs: whether to add input tensor to output
    """
    token_nums, slice_size = inputs.shape
    assert lora_b_weights.ndim in [3, 4]
    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        rank = lora_b_weights.size(-1)
    else:
        rank = lora_b_weights.size(-2)
    max_seq_length = token_nums
    batches = 2
    slice_offset = 0
    add_inputs = False
    device = inputs.device
    dtype = inputs.dtype
    # Create output tensor
    output_tensor = torch.zeros(token_nums, slice_size, dtype=dtype, device=device).contiguous()
    b_seq_start_loc = torch.tensor([0, token_nums//2], dtype=torch.int32, device=device)
    seq_len_tensor = torch.tensor([token_nums//2, token_nums//2], dtype=torch.int32, device=device)
    lora_indices_tensor = torch.tensor([0, 0], dtype=torch.int32, device=device)

    token_nums, slice_size = inputs.shape
    lora_num, N, K = lora_b_weights.shape

    for batch_idx in range(b_seq_start_loc.size(0)):
        M = seq_len_tensor[batch_idx].item()
        start = b_seq_start_loc[batch_idx].item()
        lora_idx = lora_indices_tensor[batch_idx].item()
        if lora_idx == -1:
            continue

        input_slice = inputs[start:start+M, :K]  # [M, K]
        lora_slice = lora_b_weights[lora_idx, :N, :K]  # [N, K]

        # matrix multiply
        out_slice = input_slice @ lora_slice.T  # [M, N]

        # add input if requested
        if add_inputs:
            out_slice += output_tensor[start:start+M, slice_offset:slice_offset+N]

        # write back to output
        output_tensor[start:start+M, slice_offset:slice_offset+N] = out_slice

    return output_tensor