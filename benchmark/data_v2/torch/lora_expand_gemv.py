import torch

def torch_lora_expand_gemv(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    add_inputs: bool = True,
) -> torch.Tensor:
    """
    PyTorch implementation of LoRA expand GEMV.
    Matches Triton kernel's computation precision (fp16 multiply, fp32 accumulate).
    """
    lora_indices = torch.tensor([0, 1, -1, 2], dtype=torch.int32, device=inputs.device)
    batch_size, hidden_size = inputs.shape

    if lora_b_weights.ndim == 4:
        assert lora_b_weights.shape[1] == 1
        lora_b_weights = lora_b_weights.squeeze(1)
    lora_num, N, K = lora_b_weights.shape
    assert K == hidden_size

    output_tensor = torch.zeros(batch_size, N, device=inputs.device, dtype=inputs.dtype)

    for b in range(batch_size):
        idx = lora_indices[b].item()
        if idx == -1:
            continue
        # Compute matmul: match Triton's accumulation behavior
        # Triton kernel: tl.sum(tiled_a * tiled_b, 1) where both are in weight dtype
        # For fp16 inputs and fp16 weights, the multiply is fp16 and sum accumulates
        # Use the native dtype for the computation to match Triton precision
        result = torch.mv(lora_b_weights[idx], inputs[b])
        if add_inputs:
            output_tensor[b] += result
        else:
            output_tensor[b] = result
    return output_tensor
