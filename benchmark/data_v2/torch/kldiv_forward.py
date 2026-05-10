import torch
from typing import Literal

REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]

def torch_kldiv_forward(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_target: bool = False,
    reduction: REDUCTION_LITERAL = "batchmean",
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute KL divergence y_true || y_pred elementwise.
    y_pred is expected to be in log-space if log_target=False.
    """
    assert y_pred.shape == y_true.shape, "Input shapes must match"

    # Compute in fp32 for numerical stability to match Triton kernel
    y_pred_f = y_pred.float()
    y_true_f = y_true.float()

    if not log_target:
        # Match Triton kernel: y_true_safe * (log(y_true_safe) - y_pred)
        # Clamped y_true is used for BOTH the multiplier and the log argument
        y_true_safe = y_true_f.clamp(min=eps)
        loss = y_true_safe * (torch.log(y_true_safe) - y_pred_f)
    else:
        loss = torch.exp(y_true_f) * (y_true_f - y_pred_f)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.sum() / loss.numel()
    elif reduction == "batchmean":
        return loss.sum() / y_pred.size(0)
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")


def torch_kldiv_backward(
    target: torch.Tensor,
    grad_output: torch.Tensor,
    log_target: bool = False,
) -> torch.Tensor:
    """
    Compute gradient of KL divergence w.r.t. y_pred
    """
    if not log_target:
        grad = -target
    else:
        grad = -torch.exp(target)

    if torch.all(grad_output == 1.0):
        return grad
    return grad * grad_output
