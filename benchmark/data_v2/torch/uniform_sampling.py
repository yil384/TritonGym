import torch
import functools
import operator
from typing import Tuple

Shape = Tuple[int]

def volume(shape: Shape) -> int:
    return functools.reduce(operator.mul, shape, 1)

UNROLL = 4

def torch_uniform_sampling(tensor: torch.Tensor, from_: float = 0.0, to: float = 1.0, *, generator=None):
    """
    Torch version of Triton uniform sampling.
    Fills the input tensor with uniform samples in [from_, to_).
    """
    if tensor.numel() == 0:
        return tensor

    if generator is None:
        generator = torch.Generator(device=tensor.device)

    # Fill tensor with uniform values
    tensor.uniform_(from_, to, generator=generator)
    return tensor
