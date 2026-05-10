import torch
import math

def torch_argmax(inp: torch.Tensor, dim=None, keepdim=False):
    """
    Torch version of Triton argmax.
    Supports global argmax (dim=None) and along a specific dim.
    """
    if dim is None:
        # Flatten tensor and take argmax
        flat_inp = inp.view(-1)
        out_index = torch.argmax(flat_inp)
        if keepdim:
            out_shape = [1 for _ in inp.shape]
            return out_index.view(out_shape)
        else:
            return out_index
    else:
        # argmax along a specific dim
        return torch.argmax(inp, dim=dim, keepdim=keepdim)
