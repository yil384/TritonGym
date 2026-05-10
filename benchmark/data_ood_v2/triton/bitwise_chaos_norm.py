import torch
import triton
import triton.language as tl

@triton.jit
def _algorithmic_chaos_norm_kernel(
    Y_ptr,
    X_ptr,
    W1_ptr,
    W2_ptr,
    B_ptr,
    Y_row_stride,
    X_row_stride,
    n_cols,
    base_eps,
    max_val_multiplier,
    log_addend,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr_curr = X_ptr + row_idx * X_row_stride
    Y_ptr_curr = Y_ptr + row_idx * Y_row_stride

    x_row = tl.load(X_ptr_curr + col_offsets, mask=mask, other=0.0)
    w1_row = tl.load(W1_ptr + col_offsets, mask=mask, other=0.0)
    w2_row = tl.load(W2_ptr + col_offsets, mask=mask, other=0.0)
    b_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)

    max_val = tl.max(x_row, axis=0)
    dynamic_eps = base_eps + tl.abs(max_val) * max_val_multiplier

    mean = tl.sum(x_row, axis=0) / n_cols
    var = tl.sum((x_row - mean) * (x_row - mean), axis=0) / n_cols
    rstd = tl.rsqrt(var + dynamic_eps)

    normalized_row = (x_row - mean) * rstd

    x_int = x_row.to(tl.int32)
    w_selected = tl.where((x_int & 1) == 0, w1_row, w2_row)

    scaled_row = normalized_row * w_selected + b_row

    is_even_col = (col_offsets % 2) == 0
    y_row = tl.where(
        is_even_col,
        tl.log(tl.abs(scaled_row) + log_addend),
        -scaled_row
    )

    tl.store(Y_ptr_curr + col_offsets, y_row, mask=mask)


def triton_algorithmic_chaos_norm(x, w1, w2, b, base_eps = 1e-5, max_val_multiplier = 1e-6, log_addend = 1.0):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _algorithmic_chaos_norm_kernel[grid](
        Y_ptr=y,
        X_ptr=x,
        W1_ptr=w1,
        W2_ptr=w2,
        B_ptr=b,
        Y_row_stride=y.stride(0),
        X_row_stride=x.stride(0),
        n_cols=n_cols,
        base_eps=base_eps,
        max_val_multiplier=max_val_multiplier,
        log_addend=log_addend,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
