"""
Introduction to Gluon
=====================

Gluon is a GPU programming language based on the same compiler stack as Triton.
But unlike Triton, Gluon is a lower-level language that gives the user more
control and responsibility when implementing kernels.

This tutorial series covers GPU kernel development in Gluon, from the basics to
advanced optimization techniques and modern GPU hardware features, culminating
in building an efficient GEMM kernel. Basic familiarity with Triton is assumed.

At a high level, Gluon and Triton share many similarities. Both implement a
tile-based SPMD programming model, where tiles represent N-dimensional arrays
distributed over a "program". Both are Python DSLs sharing the same frontend
and JIT infrastructure.

Triton, however, abstracts many details of implementing kernels and GPU hardware
from the user. It defers to the compiler to manage tile layouts, memory
allocation, data movement, and asynchronity.

Getting these details right is important to kernel performance. While the Triton
compiler does a good job of generating efficient code for a wide range of
kernels, it can be beaten by hand-tuned low-level code. When this happens,
there is little the user can do to significantly improve performance since all
the details are hidden.

In Gluon, these details are exposed to the user. This means writing Gluon
kernels requires a deeper understanding of GPU hardware and the many aspects of
GPU programming, but it also enables writing more performant kernels by finely
controlling these low-level details.
"""

# %%
# Let's define a Gluon kernel and write its launcher. Use the `@gluon.jit`
# decorator to declare a Gluon kernel, and it can be invoked from Python with
# the same interface as a Triton kernel.

import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# %%
# We illustrate this with a trivial kernel that copies a scalar.


@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)


# %%
# The launcher is host-side code that invokes the kernel. PyTorch tensors are
# converted to global memory pointers when passed to Gluon kernels, just like in
# Triton. And the grid is specified in the same way.


def triton_gluon_copy_scalar(input):
    output = torch.empty_like(input, device=input.device)
    # Launch a single program.
    grid = (1, )
    copy_scalar_kernel[grid](input, output, num_warps=1)
    return output