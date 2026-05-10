import torch
import triton
import triton.language as tl

@triton.jit
def _triton_wormhole_connect_norm_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr, W_stride,
    B_ptr, B_stride,
    n_rows, n_cols,
    eps: tl.constexpr,
    perturbation_factor: tl.constexpr,
    divisor_guard: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    neighbor_row_idx = (row_idx + 1) % n_rows
    neighbor_col_idx = row_idx % n_cols
    wormhole_ptr = X_ptr + neighbor_row_idx * X_row_stride + neighbor_col_idx
    wormhole_val = tl.load(wormhole_ptr).to(tl.float32)

    X_ptr_curr = X_ptr + row_idx * X_row_stride
    x_row = tl.load(X_ptr_curr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w_row = tl.load(W_ptr + col_offsets * W_stride, mask=mask, other=0.0).to(tl.float32)
    b_row = tl.load(B_ptr + col_offsets * B_stride, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x_row, axis=0) / n_cols
    perturbed_mean = mean + wormhole_val * perturbation_factor
    var = tl.sum((x_row - perturbed_mean) * (x_row - perturbed_mean), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    normalized_row = (x_row - perturbed_mean) * rstd * w_row + b_row
    y_row = tl.where(
        wormhole_val > 0.5,
        normalized_row * wormhole_val,
        normalized_row / (wormhole_val + divisor_guard)
    )

    Y_ptr_curr = Y_ptr + row_idx * Y_row_stride
    tl.store(Y_ptr_curr + col_offsets, y_row, mask=mask)

def triton_wormhole_connect_norm(x, w, b, eps=1e-5, perturbation_factor=0.1, divisor_guard=1e-6):
    n_rows, n_cols = x.shape
    assert x.is_cuda and x.is_contiguous()
    assert w.is_cuda and b.is_cuda

    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)

    _triton_wormhole_connect_norm_kernel[grid](
        y, y.stride(0),
        x, x.stride(0),
        w, w.stride(0),
        b, b.stride(0),
        n_rows, n_cols,
        eps=eps,
        perturbation_factor=perturbation_factor,
        divisor_guard=divisor_guard,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y
