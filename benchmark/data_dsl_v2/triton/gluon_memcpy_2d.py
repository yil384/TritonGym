
import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def memcpy_2d_kernel(in_ptr, out_ptr,  #
                     xnumel, ynumel, xstride_in, ystride_in, xstride_out, ystride_out,  #
                     layout: gl.constexpr, XBLOCK: gl.constexpr, YBLOCK: gl.constexpr):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)

    start_x = pid_x * XBLOCK
    start_y = pid_y * YBLOCK
    # For the 1D indices, use a SliceLayout along the dimensions we will expand.
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    # expand_dims along the slice dimension returns a tensor with the parent
    # layout, so this yields [XBLOCK, 1] and [1, YBLOCK] tensors with the same
    # layout which can be broadcasted together to [XBLOCK, YBLOCK].
    in_offsets = xstride_in * indices_x[:, None] + ystride_in * indices_y[None, :]
    out_offsets = xstride_out * indices_x[:, None] + ystride_out * indices_y[None, :]

    # Compute the mask the same way: select for indices along each dimension
    # that are in bounds and broadcast them together.
    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)

    value = gl.load(in_ptr + in_offsets, mask=mask)
    gl.store(out_ptr + out_offsets, value, mask=mask)


def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)

def triton_gluon_memcpy_2d(input, XBLOCK=1024, YBLOCK=16, layout=gl.BlockedLayout([1, 1], [32, 1], [1, 1], [0, 1]), num_warps=1):
    output = torch.empty_like(input, device=input.device)
    xnumel, ynumel = input.shape
    grid = (triton.cdiv(xnumel, XBLOCK), triton.cdiv(ynumel, YBLOCK))
    # Pass the strides of the input and output tensors into the kernel. The
    # compiler will specialize the kernel if any of the strides are 1, which is
    # common for the inner dimension of tensors.
    compiled_kernel = memcpy_2d_kernel[grid](  #
        input, output, xnumel, ynumel,  #
        *input.stride(), *output.stride(),  #
        layout, XBLOCK, YBLOCK, num_warps=num_warps)
    return output
