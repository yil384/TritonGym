# benchmark/data/torch/cos_func.py
import torch

def torch_cosine_compute(A: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of the Triton cosine kernel.
    Computes element-wise cosine of the input tensor A.
    """
    return torch.cos(A)
