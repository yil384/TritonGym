import torch

def torch_quantum_fluctuation_norm(X, W, B, eps=1e-5, perturbation_factor=0.1):
    """
    PyTorch reference implementation of Quantum Fluctuation Norm.
    """
    n_cols = X.shape[1]
    col_offsets = torch.arange(0, n_cols, device=X.device, dtype=X.dtype)
    perturbed_x = X + torch.sin(col_offsets) * perturbation_factor
    chaos_mean = torch.mean(perturbed_x, dim=1, keepdim=True)
    chaos_var = torch.mean(torch.cos(X) * torch.sin(chaos_mean), dim=1, keepdim=True)
    instability_factor = torch.rsqrt(torch.abs(chaos_var) + eps)
    normalized_row = torch.tanh((X - chaos_mean) * instability_factor)
    y = torch.pow(W, 2) * normalized_row + torch.cos(B)
    return y
