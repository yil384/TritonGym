import torch
import triton
import triton.language as tl

@triton.jit
def kernel(
    M,
    Out,
    matrix_stridex,
    matrix_stridey,
    out_stridex,
    out_stridey,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    # transpose
    matrix_ptr = M + d_head_arange[None, :] * matrix_stridey + size_m_arange[:, None] * matrix_stridex
    out_ptr = Out + d_head_arange[None, :] * out_stridex + size_m_arange[:, None] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, matrix)

def triton_matrix_transpose(matrix):
    size_m, d_head = matrix.shape
    out = torch.zeros((d_head, size_m), dtype=matrix.dtype, device=matrix.device)

    grid = (1,)
    kernel[grid](
        matrix,
        out,
        *matrix.stride(),
        *out.stride(),
        size_m,
        d_head,
    )
    return out



##################################################################################################################################################


import torch

def test_triton_vs_torch():
    results = {}

#     # Test case 1: Basic matrix transpose (small matrix)
#     size_m, d_head = 16, 16
#     out = triton_matrix_transpose(size_m, d_head)
#     results["test_case_1"] = out.clone()

#     # Test case 2: Non-square matrix
#     size_m, d_head = 32, 64
#     out = triton_matrix_transpose(size_m, d_head)
#     results["test_case_2"] = out.clone()

#     return results


# # Run tests
# result_gold = test_triton_vs_torch()
# print(result_gold)