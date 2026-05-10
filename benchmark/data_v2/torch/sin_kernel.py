import torch

def torch_sin_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise sin using PyTorch, supports fp16/fp32 on CUDA.
    """
    # Ensure dtype matches the original input to avoid fp16 errors in Triton
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    out = torch.sin(x_fp32)
    return out.to(orig_dtype)
