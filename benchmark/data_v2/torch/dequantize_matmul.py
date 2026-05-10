import torch

def torch_dequantize_matmul(a: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """
    Simulate dequantize int8 matmul using float16 inputs.
    a: float16 [M, K]
    b: float16 [K, N] - simulates int8 by quantizing internally
    b_scale: float16 [N] - used as per-column scale
    """
    # Quantize b to int8 range, then dequantize
    b_max = b.abs().max(dim=0).values.clamp(min=1e-7)
    b_int8_sim = (b / b_max.unsqueeze(0) * 127.0).round().clamp(-128, 127)
    
    # Use the provided b_scale (take abs to ensure positive scale)
    scale = b_scale.abs().clamp(min=1e-7)
    
    # Dequantize: b_float = b_int8 * scale
    fp_b = b_int8_sim.float() * (scale.float().view(1, -1) / 127.0)
    
    # Matmul
    return torch.mm(a.float(), fp_b).to(a.dtype)
