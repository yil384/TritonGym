import torch
import triton
import triton.language as tl
import math

@triton.jit(do_not_specialize=["eps"])
def triton_l4_norm_kernel(
    Y, X, W,
    y_stride_r, y_stride_c,
    x_stride_r, x_stride_c,
    N, eps, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols * x_stride_c, mask=mask, other=0.0).to(tl.float32)

    # L4-Norm
    sum4 = tl.sum(x * x * x * x, axis=0)
    norm4_val = tl.sqrt(tl.sqrt(sum4 + eps))
    r_norm4 = 1 / norm4_val

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (x * r_norm4).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)

class TritonL4Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)
        BLOCK_SIZE = triton.next_power_of_2(N)

        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)

        with torch.cuda.device(x.device):
            triton_l4_norm_kernel[(M,)](
                y, x, weight,
                y.stride(0), y.stride(1),
                x.stride(0), x.stride(1),
                N, eps, BLOCK_SIZE=BLOCK_SIZE
            )
        return y

def triton_l4_norm(x, weight, eps=1e-5):
    normalized_shape = (x.size(-1),)
    return TritonL4Norm.apply(x, normalized_shape, weight, eps)
