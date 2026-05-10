import torch

def torch_matmul_leakyrelu_fp8(a: torch.Tensor, b: torch.Tensor, activation: str = "") -> torch.Tensor:
    """
    Pure PyTorch version of the Triton matmul + optional leaky ReLU.
    The Triton kernel accumulates in fp32, applies activation in fp32, then casts to fp16.

    Args:
        a: Tensor of shape (M, K)
        b: Tensor of shape (K, N)
        activation: "" or "leaky_relu"

    Returns:
        c: Tensor of shape (M, N) in float16
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # Use native matmul which matches Triton's tensor core accumulation pattern
    c = (a @ b).float()

    if activation == "leaky_relu":
        c = torch.where(c >= 0, c, 0.01 * c)

    # Cast to fp16 to match Triton kernel output
    return c.to(torch.float16)
