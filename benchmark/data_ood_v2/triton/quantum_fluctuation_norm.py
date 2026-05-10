import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def triton_quantum_fluctuation_norm_kernel(
    Y_ptr, Y_row_stride, X_ptr, X_row_stride,
    W_ptr, W_row_stride,
    B_ptr, B_row_stride,
    Chaos_A_ptr, Chaos_B_ptr, n_cols, eps,
    perturbation_factor: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    W_ptr += row_idx * W_row_stride
    B_ptr += row_idx * B_row_stride
    Chaos_A_ptr += row_idx
    Chaos_B_ptr += row_idx

    x_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
    w_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    b_row = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)

    perturbed_x = x_row + tl.sin(col_offsets.to(tl.float32)) * perturbation_factor
    chaos_mean = tl.sum(perturbed_x, axis=0) / n_cols
    chaos_var = tl.sum(tl.cos(x_row) * tl.sin(chaos_mean), axis=0) / n_cols
    instability_factor = tl.rsqrt(tl.abs(chaos_var) + eps)
    tl.store(Chaos_A_ptr, chaos_mean)
    tl.store(Chaos_B_ptr, instability_factor)

    normalized_row = libdevice.tanh((x_row - chaos_mean) * instability_factor)
    y_row = libdevice.pow(w_row, 2) * normalized_row + tl.cos(b_row)
    tl.store(Y_ptr + col_offsets, y_row, mask=mask)

def triton_quantum_fluctuation_norm(X, W, B, eps=1e-5, perturbation_factor=0.1):
    N_ROWS, N_COLS = X.shape
    Y = torch.empty_like(X)
    Chaos_A = torch.empty((N_ROWS,), device=X.device, dtype=X.dtype)
    Chaos_B = torch.empty((N_ROWS,), device=X.device, dtype=X.dtype)
    BLOCK_SIZE = triton.next_power_of_2(N_COLS)
    grid = (N_ROWS,)
    triton_quantum_fluctuation_norm_kernel[grid](
        Y, Y.stride(0),
        X, X.stride(0),
        W, W.stride(0),
        B, B.stride(0),
        Chaos_A, Chaos_B,
        N_COLS, eps,
        perturbation_factor=perturbation_factor,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return Y
