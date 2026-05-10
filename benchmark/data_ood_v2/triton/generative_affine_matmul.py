# benchmark/data/triton/generative_affine_matmul.py
import torch
import triton
import triton.language as tl

@triton.jit
def generative_affine_matmul_kernel(
        a_ptr, c_ptr,
        M, N, K,
        alpha, beta,
        stride_am, stride_ak,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k_loop = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_loop[None, :] * stride_ak)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k_loop[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        offs_k_b = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        b_generated = (offs_k_b[:, None] * alpha + offs_bn[None, :] * beta)
        b_mask = (offs_k_b[:, None] < K) & (offs_bn[None, :] < N)
        b = tl.where(b_mask, b_generated, 0.0).to(a.dtype)

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_generative_affine_matmul(a, alpha=0.01, beta=0.02):
    M, K = a.shape
    N = K
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256), )
    generative_affine_matmul_kernel[grid](
        a, c, M, N, K, alpha, beta,
        a.stride(0), a.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_warps=8, num_stages=3
    )
    return c
