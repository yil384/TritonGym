import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# %%
# This is a helper for toggling specific parts of the tutorial. Run the tutorial
# with `python 02-layouts.py` to run everything, but you can select specific
# parts with `python 02-layouts.py R_vs_throughput,LDG_STG_instructions`.


def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")


# %%
# Parameterize the kernel over the layout so we can test different layouts. Each
# program copies a block of data, but we will use the layout to distribute
# the work over all the threads.


@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # The main difference between writing this kernel in Triton and Gluon is
    # we need to specify the layout of the 1D tensor. Layouts are propagated
    # forwards through type inference, so we only need to specify the layout for
    # the indices tensor.
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel

    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)


def triton_gluon_memcpy_1d(input, XBLOCK=1024, layout=gl.BlockedLayout([2], [32], [4], [0]), num_warps=4):
    output = torch.empty_like(input, device=input.device)
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)
    return output
