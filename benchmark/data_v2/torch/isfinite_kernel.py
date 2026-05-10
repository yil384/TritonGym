import torch

def torch_isfinite_kernel(in0: torch.Tensor):
    """
    Pure PyTorch implementation of Triton isfinite kernel.
    Works element-wise for float32 and float64 tensors.
    """
    out0 = torch.empty_like(in0, dtype=torch.bool)
    assert in0.shape == out0.shape, 'operand shapes mismatch'
    # Use PyTorch isfinite directly
    out0.copy_(torch.isfinite(in0))
    return out0.to(in0.dtype)  # Convert bool to float (1.0 for True, 0.0 for False)
