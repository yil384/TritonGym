import torch

def torch_fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3):
    """
    Torch implementation of fused native layer norm without welford.
    Matches triton signature: triton_fused_native_layer_norm_no_welford(primals_1, primals_2, primals_3)
    Args:
        primals_1: weight (gamma), shape [D]
        primals_2: bias (beta), shape [D]
        primals_3: input tensor x, shape [S, D]
    Returns:
        y: layer-normalized output, shape [S, D] bfloat16
    """
    weight = primals_1
    bias = primals_2
    x = primals_3
    eps = 1e-5

    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    bias_fp32 = bias.to(torch.float32)

    # Compute mean over features (dim=1)
    mean = x_fp32.mean(dim=1, keepdim=True)
    # Compute variance over features
    var = ((x_fp32 - mean) ** 2).mean(dim=1, keepdim=True)
    # Reciprocal of std
    rstd = torch.rsqrt(var + eps)
    # Normalize
    x_hat = (x_fp32 - mean) * rstd
    # Apply weight and bias
    y = x_hat * weight_fp32 + bias_fp32
    y = y.to(torch.bfloat16)
    return y
