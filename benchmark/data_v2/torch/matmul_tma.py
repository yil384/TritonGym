import torch

def torch_matmul_tma(a, b, NUM_CTAS=1, NUM_WARPS=4, TRANS_A=False, TRANS_B=False, OUTPUT_F16=False):
    """
    PyTorch implementation of Triton matmul TMA kernel.
    Computes c = a @ b with optional transposition and output dtype control.
    """
    c = torch.matmul(a, b)
    if OUTPUT_F16:
        c = c.to(torch.float16)
    else:
        c = c.float()
    return c
