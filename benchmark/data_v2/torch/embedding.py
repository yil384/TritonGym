import torch

def torch_embedding(weight: torch.Tensor, vob_start_id: int=10, vob_end_id: int=1000):
    """
    Torch version of embedding operator.
    Matches the Triton kernel which:
      1. Generates random input_ids in [vob_start_id, vob_end_id)
      2. For each token, subtracts vob_start_id to get weight index
      3. Loads from weight[index] if id is in valid range, else stores 0

    Note: Since both triton and torch generate random input_ids internally,
    outputs will NOT match between them unless the same input_ids are used.
    """
    sequence_length = weight.shape[0]
    embedding_dim = weight.shape[1]
    input_ids = torch.randint(vob_start_id, vob_end_id, (sequence_length,), dtype=torch.int32, device=weight.device)
    out = torch.zeros(sequence_length, embedding_dim, dtype=weight.dtype, device=weight.device)

    mask = (input_ids >= vob_start_id) & (input_ids < vob_end_id)
    valid_ids = input_ids.clone()
    valid_ids[mask] = valid_ids[mask] - vob_start_id
    valid_ids[~mask] = 0

    out[mask] = weight[valid_ids[mask]]

    return out
