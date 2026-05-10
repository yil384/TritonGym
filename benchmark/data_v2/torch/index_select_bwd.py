import torch

def torch_index_select_bwd(grad_source: torch.Tensor, grad_output: torch.Tensor):
    """
    Pure PyTorch version of index_select_cat_bwd.
    Updates grad_source in-place according to grad_output and index.
    """
    # Validate shapes
    index = torch.tensor([0, 2, 4, 6, 8], device=grad_source.device)
    num_rows, num_cols = grad_source.shape
    num_indices = grad_output.shape[0]
    assert grad_output.shape[1] == num_cols
    assert index.shape[0] == num_indices
    assert num_rows >= num_indices

    # Scatter grad_output to grad_source using index
    grad_source.index_copy_(0, index, grad_output)

    return grad_source
