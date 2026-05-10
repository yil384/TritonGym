# benchmark/data/torch/rowwise_quantization.py
import torch

def torch_rowwise_quantization(x: torch.Tensor):
    """
    PyTorch implementation of Triton rowwise quantization.
    Each row is scaled to int8 using its max absolute value.
    Returns quantized tensor and row-wise max values.
    """
    assert x.dim() == 2, "Only 2D tensors are supported"
    rows, cols = x.shape
    output = torch.empty_like(x, dtype=torch.int8)
    output_maxs = torch.empty(rows, device=x.device, dtype=torch.float16)

    # Compute row-wise max abs
    row_max = x.abs().amax(dim=1)
    output_maxs.copy_(row_max)

    # Avoid division by zero
    row_max_safe = row_max.clone()
    row_max_safe[row_max_safe == 0] = 1.0

    # Scale and quantize to int8
    scaled = x / row_max_safe[:, None] * 127
    output.copy_(scaled.round().clamp(-128, 127).to(torch.int8))
    return output
