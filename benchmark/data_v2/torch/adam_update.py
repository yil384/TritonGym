import torch

def torch_adam_update(p, grad, exp_avg, lr=0.01, wd=0.01, beta1=0.9, beta2=0.999):
    """
    PyTorch implementation of adam_update matching the Triton kernel.
    The Triton kernel modifies p and exp_avg in-place.
    """
    # Stepweight decay
    p.mul_(1 - lr * wd)
    # Diff between momentum running average and grad
    diff = exp_avg - grad
    # Weight update
    update = diff * beta1 + grad
    # torch.sign
    can_update = (update != 0).to(p.dtype)
    update_sign = torch.where(update > 0, -lr, lr)
    p.add_(update_sign * can_update)
    # Decay the momentum running average coefficient
    exp_avg.copy_(diff * beta2 + grad)
    return p