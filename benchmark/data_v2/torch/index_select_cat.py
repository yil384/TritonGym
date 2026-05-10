import torch

def torch_index_select_cat(output: torch.Tensor, source: torch.Tensor):
    """
    Pure PyTorch version of index_select_cat.
    Copies rows from source to output according to index.
    Generates valid indices based on source dimensions.
    """
    if source.ndim != 2:
        raise ValueError(f"Expected 2-dimensional source tensor, got {source.ndim}.")

    num_rows, num_cols = source.shape
    num_output_rows = output.shape[0]

    # Generate valid indices within bounds of source rows
    num_indices = min(num_output_rows, num_rows)
    index = torch.arange(num_indices, device=source.device)

    # Copy selected rows from source into output
    output[:num_indices] = source[index]

    return output
