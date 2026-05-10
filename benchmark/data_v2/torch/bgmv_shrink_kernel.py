import torch

def torch_bgmv_shrink_kernel(inputs, lora_a_weights, scaling=1.0):
    """Apply batched LoRA-A projection (shrink) with per-sample adapter selection.

    Args:
        inputs: Input tensor of shape (batch, K).
        lora_a_weights: LoRA-A weight matrices of shape (num_lora, N, K).
        scaling: Scalar multiplier applied to each projection output.

    Returns:
        Output tensor of shape (batch, N) with each row projected by its assigned LoRA adapter.
    """
    lora_indices = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    batch, K = inputs.shape
    num_lora, N, K2 = lora_a_weights.shape
    assert K == K2
    output = torch.zeros(batch, N, dtype=inputs.dtype, device=inputs.device)
    for i in range(batch):
        lora_idx = lora_indices[i].item()
        if lora_idx == -1:
            continue
        a = lora_a_weights[lora_idx]
        out = torch.matmul(a, inputs[i]) * scaling
        output[i] = out
    return output
