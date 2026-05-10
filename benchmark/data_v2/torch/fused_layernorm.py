import torch

def torch_fused_layernorm(primals_1: torch.Tensor, primals_2: torch.Tensor, primals_3: torch.Tensor):
    """
    Pure PyTorch implementation of Triton fused native layer norm.
    Matches triton signature: triton_fused_layernorm(primals_1, primals_2, primals_3)
    Args:
        primals_1: weight [D] bfloat16
        primals_2: bias [D] bfloat16
        primals_3: input x [S, D] bfloat16
    Returns:
        out: [S, D] bfloat16
    """
    weight = primals_1
    bias = primals_2
    x = primals_3
    eps = 1e-5

    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    bias_fp32 = bias.to(torch.float32)

    mean = x_fp32.mean(dim=1, keepdim=True)
    var = x_fp32.var(dim=1, unbiased=False, keepdim=True)
    inv_std = torch.rsqrt(var + eps)

    normalized = (x_fp32 - mean) * inv_std
    out = normalized * weight_fp32 + bias_fp32
    out = out.to(torch.bfloat16)

    return out
