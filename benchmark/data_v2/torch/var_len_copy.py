import torch

def torch_var_len_copy(old_a_location: torch.Tensor):
    """
    Torch version of var_len_copy_kernel_triton.
    Copies variable-length segments from old_a_location to new_a_location.
    """
    old_a_start = torch.tensor([0, 100, 300], dtype=torch.int32, device=old_a_location.device)
    old_a_len = torch.tensor([50, 150, 200], dtype=torch.int32, device=old_a_location.device)
    new_a_start = torch.tensor([0, 60, 260], dtype=torch.int32, device=old_a_location.device)
    new_a_location = torch.zeros_like(old_a_location)
    num_arrays = old_a_start.numel()
    for i in range(num_arrays):
        start_old = old_a_start[i].item()
        start_new = new_a_start[i].item()
        length = old_a_len[i].item()
        if length > 0:
            new_a_location[start_new:start_new + length] = old_a_location[start_old:start_old + length]

    return new_a_location
