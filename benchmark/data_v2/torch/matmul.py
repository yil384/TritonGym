import torch

def torch_matmul(a: torch.Tensor, b: torch.Tensor, transpose_a: bool = False, transpose_b: bool = False) -> torch.Tensor:
    """
    PyTorch matmul matching the Triton kernel which accumulates in fp32
    then casts output to fp16.
    """
    if transpose_a:
        a = a.transpose(-2, -1)
    if transpose_b:
        b = b.transpose(-2, -1)
    # Use native matmul which matches Triton's tensor core accumulation pattern
    c = torch.matmul(a, b)
    return c.to(torch.float16)
