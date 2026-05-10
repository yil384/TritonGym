# triton_implementation.py
import torch
import triton
import triton.language as tl

@triton.jit
def soft_sorting_index_kernel(
    A_ptr,      # pointer to input tensor A
    C_ptr,      # pointer to output tensor C
    M,          # number of rows of A
    N,          # number of columns of A
    stride_am,  # row stride of A
    stride_an,  # column stride of A
    stride_cm,  # row stride of C
    stride_cn,  # column stride of C
    BLOCK_SIZE_N: tl.constexpr, # Triton tile size
):
    """
    Triton kernel implementing the "soft" sorting index algorithm.
    Each program instance processes one row.
    """
    # Get the row index that the current program instance is processing
    pid_m = tl.program_id(axis=0)
    
    # --- Pass 1: compute the min and max of the current row ---
    row_start_ptr_a = A_ptr + pid_m * stride_am
    
    # Initialize row-level min and max
    row_min_val = float('inf')
    row_max_val = float('-inf')
    
    # Iterate over the entire row in steps of BLOCK_SIZE_N
    for offset in range(0, N, BLOCK_SIZE_N):
        # Compute the column indices for the current block
        cols = offset + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        # Safely load the data block
        # For min, set out-of-bounds values to inf; for max, set to -inf
        a_for_min = tl.load(row_start_ptr_a + cols, mask=mask, other=float('inf'))
        a_for_max = tl.load(row_start_ptr_a + cols, mask=mask, other=float('-inf'))
        
        # Update the row-level min and max
        row_min_val = tl.minimum(row_min_val, tl.min(a_for_min, axis=0))
        row_max_val = tl.maximum(row_max_val, tl.max(a_for_max, axis=0))
        
    # --- Compute the range and handle edge cases ---
    range_val = row_max_val - row_min_val
    if range_val == 0:
        range_val = 1.0
        
    # --- Pass 2: compute and store results ---
    row_start_ptr_c = C_ptr + pid_m * stride_cm
    
    for offset in range(0, N, BLOCK_SIZE_N):
        cols = offset + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        # Reload the data
        a = tl.load(row_start_ptr_a + cols, mask=mask, other=0)
        
        # Compute score
        score = (a - row_min_val) / range_val
        
        # Compute j (column index)
        j = cols.to(tl.float32)
        
        # Compute the final value
        c = j * score + (N - 1 - j) * (1 - score)
        
        # Write back the results
        tl.store(row_start_ptr_c + cols, c, mask=mask)


def soft_sorting_index_triton(A: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the soft_sorting_index_kernel.
    """
    if A.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional")
    if not A.is_cuda:
        raise TypeError("Input tensor must be on a CUDA device")

    M, N = A.shape
    
    # Create output tensor
    C = torch.empty_like(A)
    
    # Configure the Triton kernel launch grid
    # Each program instance processes one row, so grid is (M,)
    grid = (M,)
    
    # Choose an appropriate block size
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    if BLOCK_SIZE_N > 2048: # Adjust based on GPU capability
        BLOCK_SIZE_N = 2048

    # Launch the kernel
    soft_sorting_index_kernel[grid](
        A, C,
        M, N,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return C


import torch
# torch_reference.py
import torch

def soft_sorting_index_pytorch(A: torch.Tensor) -> torch.Tensor:
    """
    Implements the "soft" sorting index algorithm using PyTorch.

    Args:
        A (torch.Tensor): A 2-dimensional input tensor with shape (M, N).

    Returns:
        torch.Tensor: The computed "soft" sorting index matrix, same shape as A.
    """
    # Ensure the input is 2-dimensional
    if A.dim() != 2:
        raise ValueError("Input tensor must be 2-dimensional")
        
    M, N = A.shape
    
    # 1. Find the max and min of each row
    # keepdim=True is for subsequent broadcasting operations
    min_val, _ = torch.min(A, dim=1, keepdim=True)
    max_val, _ = torch.max(A, dim=1, keepdim=True)
    
    # 2. Compute the range, and handle the edge case where the denominator is zero (all elements in a row are equal)
    range_val = max_val - min_val
    # If range_val is 0, set it to 1.0 to avoid division by zero.
    # In this case, (A - min_val) is also 0, so score will correctly be 0.
    range_val = torch.where(range_val == 0, 1.0, range_val)
    
    # 3. Compute the normalized score
    # Using broadcasting, subtract the (M, 1) tensor from the (M, N) tensor
    score = (A - min_val) / range_val
    
    # 4. Compute the weighted index
    # Create a column index matrix of shape (M, N) with values [0, 1, 2, ..., N-1]
    j = torch.arange(N, device=A.device, dtype=A.dtype).expand_as(A)
    
    # Compute the final result according to the formula
    # C[i, j] = j * score[j] + (N - 1 - j) * (1 - score[j])
    C = j * score + (N - 1 - j) * (1 - score)
    
    return C

def run_test():
    """
    Run tests to compare the correctness of PyTorch and Triton implementations.
    """
    # --- Test configuration ---
    M, N = 4096, 4096
    DTYPE = torch.float32
    DEVICE = 'cuda'

    print("="*50)
    print(f"Test configuration:")
    print(f"  - Shape (M, N): ({M}, {N})")
    print(f"  - Data type: {DTYPE}")
    print(f"  - Device: {DEVICE}")
    print("="*50)

    # --- Standard test ---
    print("\n[Standard test] Testing with random data...")
    # Create random input tensor
    input_tensor = torch.randn((M, N), device=DEVICE, dtype=DTYPE)

    # Compute using PyTorch
    output_torch = soft_sorting_index_pytorch(input_tensor)

    # Compute using Triton
    output_triton = soft_sorting_index_triton(input_tensor)

    # Compare results
    max_diff = (output_torch - output_triton).abs().max().item()

    print(f"diff: {output_torch - output_triton}")
    print(f"Maximum absolute difference: {max_diff:.6f}")