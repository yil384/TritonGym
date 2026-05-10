# benchmark/data/torch/cross_entropy_ops.py
import torch
from typing import Tuple

def torch_cross_entropy_ops(
    logits: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignored_index: int = -100,
    inplace_backward: bool = False,
    process_group=None,
) -> torch.Tensor:
    """
    PyTorch version of the Triton cross-entropy kernel.
    Supports label smoothing, ignored index, logit scaling, and LSE square regularization.
    Matches triton_cross_entropy_ops signature (7 args).
    inplace_backward and process_group are ignored in torch impl.
    """
    logits = logits * logit_scale
    n_rows, n_cols = logits.shape
    torch.manual_seed(42)
    labels = torch.randint(0, n_cols, (n_rows,), device=logits.device)

    # Compute log-sum-exp per row
    lse = torch.logsumexp(logits.float(), dim=1)

    # Initialize loss
    losses = torch.zeros_like(lse)

    # Compute loss
    for i in range(n_rows):
        label = labels[i]
        if label == ignored_index:
            continue
        if label_smoothing > 0.0:
            smooth_positive = 1.0 - label_smoothing
            smooth_negative = label_smoothing / n_cols
            losses[i] = -smooth_positive * logits[i, label].float() - smooth_negative * logits[i].float().sum() + lse[i]
        else:
            losses[i] = -logits[i, label].float() + lse[i]

    # Add LSE square scale if needed
    if lse_square_scale != 0.0:
        z_losses = lse_square_scale * lse.square()
        z_losses[labels == ignored_index] = 0.0
        losses = losses + z_losses

    return losses
