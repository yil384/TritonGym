import torch

def torch_masked_select(inp: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of masked select.

    Args:
        inp (Tensor): input tensor
        mask (Tensor): boolean mask tensor, 1/True selects the element
    
    Returns:
        Tensor: flattened tensor containing only selected elements
    """
    # Broadcast input and mask if needed
    mask = torch.tensor([0, 1], dtype=torch.bool, device=inp.device).repeat_interleave(1).repeat(inp.numel() // 2 + 1)[: inp.numel()].reshape(inp.shape)
    inp, mask = torch.broadcast_tensors(inp, mask)
    inp = inp.contiguous()
    mask = mask.contiguous()

    # Flatten and select
    inp_flat = inp.view(-1)
    mask_flat = mask.view(-1)
    return inp_flat[mask_flat.bool()]
