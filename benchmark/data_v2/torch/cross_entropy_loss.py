# benchmark/data/torch/cross_entropy_loss.py
import torch
from typing import Tuple

def torch_cross_entropy_loss(
    logits: torch.Tensor,
    label_smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    ignored_index: int = -100,
    inplace_backward: bool = False,
    process_group=None,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of cross-entropy loss with optional label smoothing
    and LSE square regularization.
    Matches triton_cross_entropy_loss signature (6 args).
    inplace_backward and process_group are ignored in torch impl.
    """
    device = logits.device
    n_rows, n_cols = logits.shape
    labels = torch.randint(0, n_cols, (n_rows,), device=device)

    # Compute log-sum-exp (LSE) per row
    lse = torch.logsumexp(logits.float(), dim=1)

    # One-hot labels
    one_hot = torch.zeros_like(logits, dtype=torch.float32)
    mask = (labels != ignored_index)
    one_hot[torch.arange(n_rows, device=device)[mask], labels[mask]] = 1.0

    # Apply label smoothing if needed
    if label_smoothing > 0.0:
        one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / n_cols

    # Compute negative log-likelihood
    log_probs = logits.float() - lse.unsqueeze(1)
    losses = -(one_hot * log_probs).sum(dim=1)

    # Add LSE square regularization
    if lse_square_scale != 0.0:
        z_losses = lse_square_scale * lse.square()
        z_losses[labels == ignored_index] = 0.0
        losses = losses + z_losses

    # Mask ignored index
    losses[labels == ignored_index] = 0.0

    return losses
