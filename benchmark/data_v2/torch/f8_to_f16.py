import torch

def torch_f8_to_f16(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate float8 conversion: float16 -> float8_e5m2 -> float16.
    This simulates the precision loss of f8 conversion.
    """
    # Clamp to float8_e5m2 representable range
    max_f8 = 57344.0  # max value for float8_e5m2
    x_clamped = x.float().clamp(-max_f8, max_f8)
    # Convert through float8 to simulate precision loss
    x_f8 = x_clamped.to(torch.float8_e5m2)
    return x_f8.to(torch.float16)
