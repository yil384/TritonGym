import torch

def torch_dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise dequantization of int8 tensor x using state_x as scale per row.
    x: int8 tensor of shape [B, N]
    state_x: float32 tensor of shape [B] (per-row scale)
    returns: float16 tensor [B, N]
    """
    # Convert int8 to float and scale by row-wise state
    output = x.float() * (state_x.view(-1, 1) / 127.0)
    return output.half()
