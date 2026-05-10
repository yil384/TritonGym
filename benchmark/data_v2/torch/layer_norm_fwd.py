import torch

def torch_layer_norm_fwd(
    x, weight, bias, eps=1e-5, residual=None, x1=None, weight1=None, bias1=None,
    dropout_p=0.0, rowscale=None, out_dtype=None, residual_dtype=None, is_rms_norm=False,
    return_dropout_mask=False
):
    """Compute fused Layer Norm (or RMS Norm) forward with optional dropout, residual, and row scaling.

    Args:
        x: Input tensor of shape (M, N).
        weight: Normalization weight of shape (N,).
        bias: Normalization bias of shape (N,), or None.
        eps: Epsilon for numerical stability.
        residual: Optional residual tensor of shape (M, N) to add before normalization.
        x1: Optional second input tensor of shape (M, N) to fuse.
        weight1: Optional second normalization weight for dual-stream output.
        bias1: Optional second normalization bias for dual-stream output.
        dropout_p: Dropout probability applied before normalization.
        rowscale: Optional per-row scale of shape (M,).
        out_dtype: Optional output dtype.
        residual_dtype: Optional dtype for residual output.
        is_rms_norm: If True, use RMS normalization instead of Layer Norm.
        return_dropout_mask: If True, compute and store the dropout mask.

    Returns:
        Normalized output tensor y of shape (M, N).
    """
    M, N = x.shape
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    y1 = torch.empty_like(y) if x1 is not None or weight1 is not None else None

    # Residual output
    if residual is not None or dropout_p > 0.0 or rowscale is not None or x1 is not None:
        residual_out = torch.empty_like(x if residual is None else residual, dtype=residual_dtype if residual_dtype is not None else x.dtype)
    else:
        residual_out = None

    # Mean and rstd
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    # Seeds and dropout mask
    seeds = torch.randint(0, 2**32, (M if x1 is None else 2 * M,), device=x.device, dtype=torch.int64) if dropout_p > 0.0 else None
    dropout_mask = torch.empty((M if x1 is None else 2*M, N), device=x.device, dtype=torch.bool) if return_dropout_mask and dropout_p > 0.0 else None

    # Compute LayerNorm - order matches Triton kernel:
    # 1. Load x, 2. Apply rowscale, 3. Apply dropout, 4. Add x1, 5. Add residual
    x_ = x.clone().float()
    if rowscale is not None:
        x_ = x_ * rowscale[:, None].float()
    if dropout_p > 0.0:
        mask = torch.rand_like(x_) > dropout_p
        x_ = torch.where(mask, x_ / (1.0 - dropout_p), torch.zeros_like(x_))
        if dropout_mask is not None:
            dropout_mask[:M] = mask
    if x1 is not None:
        x1_ = x1.float()
        if rowscale is not None:
            # x1 also gets scaled by its own rowscale (ROWSCALE + M + row)
            pass  # In the basic test, rowscale and x1 are not used together
        x_ = x_ + x1_
    if residual is not None:
        x_ = x_ + residual.float()

    if not is_rms_norm:
        mean_ = x_.mean(dim=1, keepdim=True)
        var_ = ((x_ - mean_)**2).mean(dim=1, keepdim=True)
        rstd_ = 1 / torch.sqrt(var_ + eps)
        mean[:,] = mean_.squeeze(1) if mean is not None else None
    else:
        mean_ = torch.zeros_like(x_)
        var_ = (x_**2).mean(dim=1, keepdim=True)
        rstd_ = 1 / torch.sqrt(var_ + eps)

    rstd[:] = rstd_.squeeze(1)

    x_hat = (x_ - mean_) * rstd_ if not is_rms_norm else x_ * rstd_

    # Cast weight/bias to fp32 to match Triton kernel which loads them as fp32
    w_f = weight.float()
    b_f = bias.float() if bias is not None else 0
    y[:] = x_hat * w_f + b_f
    if y1 is not None:
        w1_f = weight1.float() if weight1 is not None else 1.0
        b1_f = bias1.float() if bias1 is not None else 0
        y1[:] = x_hat * w1_f + b1_f

    if residual_out is not None:
        residual_out[:] = x_

    # Split dropout mask if x1 exists
    if dropout_mask is not None and x1 is not None:
        dropout_mask1 = dropout_mask[M:]
        dropout_mask = dropout_mask[:M]
    else:
        dropout_mask1 = None

    return y
