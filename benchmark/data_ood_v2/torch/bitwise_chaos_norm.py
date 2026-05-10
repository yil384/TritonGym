import torch

def torch_bitwise_chaos_norm(x, w1, w2, b, base_eps = 1e-5, max_val_multiplier = 1e-6, log_addend = 1.0):
    """
    PyTorch reference implementation of the Algorithmic Chaos Norm.
    """
    n_cols = x.shape[1]
    
    # 1. dynamic Epsilon
    max_val = torch.max(x, dim=1, keepdim=True)[0]
    dynamic_eps = base_eps + torch.abs(max_val) * max_val_multiplier
    
    # 2. Compute standard statistics
    mean = x.mean(dim=1, keepdim=True)
    var = ((x - mean)**2).mean(dim=1, keepdim=True)
    rstd = torch.rsqrt(var + dynamic_eps)
    
    normalized_x = (x - mean) * rstd
    
    # 3. Bitwise operation to select weights
    x_int = x.to(torch.int32)
    w_selected = torch.where((x_int & 1) == 0, w1, w2)
    
    scaled_x = normalized_x * w_selected + b
    
    # 4. Odd-even column splitting
    col_indices = torch.arange(n_cols, device=x.device)
    is_even_col = (col_indices % 2) == 0
    
    y = torch.where(
        is_even_col,
        torch.log(torch.abs(scaled_x) + log_addend),
        -scaled_x
    )
    
    return y
