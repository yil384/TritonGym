import torch
import math

def torch_geglu_forward(a: torch.Tensor, b: torch.Tensor):
    """
    Pure PyTorch version of GeGLU with tanh approximation.
    Returns (a, b, c) where c = GeGLU(a, b)
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)

    a_float = a.to(torch.float32)
    a_cubed = a_float * a_float * a_float
    tanh_arg = sqrt_2_over_pi * (a_float + 0.044715 * a_cubed)
    tanh_result = torch.tanh(tanh_arg)
    geglu_a = 0.5 * a_float * (1 + tanh_result)
    c = geglu_a * b
    return c

def torch_geglu_backward(a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor):
    """
    Pure PyTorch backward for GeGLU tanh variant.
    Returns (da, db)
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)

    a_float = a.to(torch.float32)
    a_cubed = a_float * a_float * a_float
    tanh_arg = sqrt_2_over_pi * (a_float + 0.044715 * a_cubed)
    tanh_result = torch.tanh(tanh_arg)
    geglu_a = 0.5 * a_float * (1 + tanh_result)

    db = dc * geglu_a

    term1 = 0.5 * (1 + tanh_result)
    term2 = 0.5 * a_float * (1 - tanh_result * tanh_result) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_float * a_float))
    da = dc * b * (term1 + term2)

    return da, db
