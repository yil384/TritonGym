def torch_tlx_dual_add(x, y, a, b):
    """Compute two independent element-wise additions in a single call.

    Args:
        x: First input tensor of shape (M, N).
        y: Second input tensor of shape (M, N), added to x.
        a: Third input tensor of shape (M, N).
        b: Fourth input tensor of shape (M, N), added to a.

    Returns:
        Tuple of (x + y, a + b), each of shape (M, N).
    """
    return x + y, a + b

if __name__ == "__main__":
    input_shapes = [
        ((1024, 1024), (1024, 1024), (1024, 1024), (1024, 1024)),
        ((4096, 4096), (4096, 4096), (4096, 4096), (4096, 4096)),  # may fail: not broadcastable
    ]