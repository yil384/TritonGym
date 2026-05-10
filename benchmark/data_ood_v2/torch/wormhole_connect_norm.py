import torch

def torch_wormhole_connect_norm(x, w, b, eps=1e-5, perturbation_factor=0.1, divisor_guard=1e-6):
    """Apply row-wise normalization with cross-row "wormhole" perturbation and conditional scaling.

    For each row, perturbs the mean using a value from a neighboring row, normalizes,
    applies affine transform, then scales or divides by the wormhole value.

    Args:
        x: Input tensor of shape (n_rows, n_cols).
        w: Affine weight of shape (n_cols,).
        b: Affine bias of shape (n_cols,).
        eps: Epsilon for numerical stability in rsqrt.
        perturbation_factor: Scale for the wormhole perturbation on the mean.
        divisor_guard: Small constant to prevent division by zero.

    Returns:
        Normalized and conditionally scaled output tensor of shape (n_rows, n_cols).
    """
    n_rows, n_cols = x.shape
    y = torch.zeros_like(x)

    for i in range(n_rows):
        neighbor_row_idx = (i + 1) % n_rows
        neighbor_col_idx = i % n_cols
        wormhole_val = x[neighbor_row_idx, neighbor_col_idx].item()

        x_row = x[i, :]
        mean = x_row.mean()
        perturbed_mean = mean + wormhole_val * perturbation_factor
        var = ((x_row - perturbed_mean) ** 2).mean()
        rstd = torch.rsqrt(var + eps)

        normalized_row = (x_row - perturbed_mean) * rstd * w + b
        if wormhole_val > 0.5:
            y_row = normalized_row * wormhole_val
        else:
            y_row = normalized_row / (wormhole_val + divisor_guard)
        y[i, :] = y_row

    return y
