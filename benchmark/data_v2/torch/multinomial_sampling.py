import torch

def torch_multinomial_sampling(
    scores: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch multinomial sampling with softmax to ensure valid probabilities.
    """
    assert scores.dim() == 2
    batch_size, num_tokens = scores.size()
    device = scores.device

    if num_tokens == 1:
        return torch.zeros(batch_size, device=device, dtype=torch.long)

    # Apply softmax to get valid probabilities from arbitrary inputs
    probs = torch.softmax(scores.float(), dim=1)
    sampled_indices = torch.multinomial(probs, num_samples=1)  # [batch, 1]
    return sampled_indices.squeeze(1)  # [batch]
