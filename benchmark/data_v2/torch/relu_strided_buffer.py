import torch
from typing import Union

def torch_relu_strided_buffer(
    in0: Union[torch.Tensor],
    out0: Union[torch.Tensor]
) -> torch.Tensor:
    """
    Pure PyTorch replacement for triton_relu_strided_buffer.
    Args:
        in0: input tensor
        out0: output tensor (same shape as in0)
    Returns:
        out0: tensor after ReLU
    """
    assert in0.shape == out0.shape, "operand shapes mismatch"
    out0.copy_(torch.clamp(in0, min=0))
    return out0
