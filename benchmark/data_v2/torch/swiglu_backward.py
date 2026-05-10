import torch

def torch_swiglu_backward(xy: torch.Tensor, dout: torch.Tensor, dxy=None, recompute_output=False, out=None):
    """
    Pure PyTorch version of Swiglu backward.
    xy: [batch, 2*hidden]
    dout: [batch, hidden]
    dxy: optional pre-allocated output tensor for gradients
    recompute_output: if True, also recompute the forward output
    out: optional pre-allocated output tensor for recomputed output
    """
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)
    dout = dout.reshape(-1, dout.shape[-1])

    assert dout.shape == x.shape, "Shape mismatch between dout and x"

    if dxy is None:
        dxy = torch.empty_like(xy)
    else:
        dxy = dxy.reshape(-1, dxy.shape[-1])
        assert dxy.shape == xy.shape

    x_sigmoid = torch.sigmoid(x.float())
    dx = (x_sigmoid * (1 + x.float() * (1 - x_sigmoid)) * y.float() * dout.float()).to(xy.dtype)
    dy = (x.float() * x_sigmoid * dout.float()).to(xy.dtype)

    dx_out, dy_out = dxy.chunk(2, dim=-1)
    dx_out.copy_(dx)
    dy_out.copy_(dy)

    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        else:
            out = out.reshape(-1, out.shape[-1])
            assert out.shape == x.shape
        out_val = (x.float() * x_sigmoid * y.float()).to(xy.dtype)
        out.copy_(out_val)
        return dxy.reshape(*batch_shape, dxy.shape[-1]), out.reshape(*batch_shape, out.shape[-1])
    else:
        return dxy.reshape(*batch_shape, dxy.shape[-1])
