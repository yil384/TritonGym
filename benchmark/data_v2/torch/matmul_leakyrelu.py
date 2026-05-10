import torch

def torch_matmul_leakyrelu(a: torch.Tensor, b: torch.Tensor, activation="leaky_relu"):
    """
    PyTorch implementation of Triton matmul kernel with optional leaky_relu activation.
    The Triton kernel accumulates in fp32, applies activation in fp32, then casts to fp16.

    Args:
        a: [M, K]
        b: [K, N]
        activation: str, either "" or "leaky_relu"
    Returns:
        c: [M, N] in float16
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"

    # Accumulate in fp32 to match Triton kernel behavior
    c = torch.matmul(a.float(), b.float())

    # Apply activation if specified (in fp32, matching Triton)
    if activation == "leaky_relu":
        c = torch.where(c >= 0, c, 0.01 * c)

    # Convert to float16 to match Triton kernel output
    c = c.to(torch.float16)
    return c
