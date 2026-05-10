import torch
def torch_tlx_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute general matrix multiplication (GEMM) of two 2D tensors.

    Args:
        a: Left input matrix of shape (M, K).
        b: Right input matrix of shape (K, N).

    Returns:
        Product matrix of shape (M, N).
    """
    return torch.matmul(a, b)

if __name__ == "__main__":
    input_shapes = [
        ((1024, 1024), (1024, 1024)),
        ((1024, 1024), (1024, 4096)),
        ((4096, 4096), (4096, 1024)),
    ]