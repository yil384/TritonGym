import torch
from typing import Optional

def torch_conv2d_fwd(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    stride_height: int = 1,
    stride_width: int = 1,
    padding_height: int = 0,
    padding_width: int = 0,
    groups: int = 1,
    fp16: bool = False,
    tf32: bool = False
) -> torch.Tensor:
    """
    Pure PyTorch implementation of 2D convolution matching the Triton kernel.
    Supports groups, padding, stride.
    """
    B, C_in, H_in, W_in = input_tensor.shape
    C_out, C_in_per_group, _, _ = weight_tensor.shape
    kernel_height = weight_tensor.shape[2]
    kernel_width = weight_tensor.shape[3]

    H_out = (H_in + 2 * padding_height - kernel_height) // stride_height + 1
    W_out = (W_in + 2 * padding_width - kernel_width) // stride_width + 1

    # Prepare output tensor
    output = torch.zeros(B, C_out, H_out, W_out, device=input_tensor.device, dtype=torch.float32)

    # Optionally cast to fp16
    if fp16:
        input_tensor = input_tensor.half()
        weight_tensor = weight_tensor.half()

    # Iterate over groups
    in_group_dim = C_in // groups
    out_group_dim = C_out // groups

    for g in range(groups):
        in_start = g * in_group_dim
        in_end = (g + 1) * in_group_dim
        out_start = g * out_group_dim
        out_end = (g + 1) * out_group_dim

        input_g = input_tensor[:, in_start:in_end, :, :]
        weight_g = weight_tensor[out_start:out_end, :, :, :]

        # Unfold input for conv
        input_unf = torch.nn.functional.unfold(
            torch.nn.functional.pad(input_g, (padding_width, padding_width, padding_height, padding_height)),
            kernel_size=(kernel_height, kernel_width),
            stride=(stride_height, stride_width)
        )  # shape: [B, C_in_per_group * kh * kw, H_out*W_out]

        weight_flat = weight_g.view(out_group_dim, -1)  # shape: [C_out_group, C_in_per_group*kh*kw]

        out_flat = weight_flat @ input_unf  # [C_out_group, H_out*W_out] per batch
        out_flat = out_flat.view(B, out_group_dim, H_out, W_out)
        output[:, out_start:out_end, :, :] = out_flat

    return output
