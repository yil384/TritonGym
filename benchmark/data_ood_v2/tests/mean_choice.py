import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
# 1. Triton Kernel Implementation
# ------------------------------------------------------------------

@triton.jit
def conditionally_weighted_aggregation_kernel(
    # Pointers to matrices
    A_ptr,
    B_ptr,
    C_ptr,
    # Matrix dimensions
    M,
    N,
    # Strides for moving between rows
    stride_am,
    stride_bm,
    stride_cm,
    # Compile-time constants
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for Conditionally Weighted Row Aggregation.
    
    For each row `i` in matrix A:
    1. Calculates the mean of A[i, :], let's call it `mean_a`.
    2. Calculates the sum of the first half of B[i, :], `sum_b_first`.
    3. Calculates the sum of the second half of B[i, :], `sum_b_second`.
    4. For each element A[i, j], the output C[i, j] is:
       - A[i, j] * sum_b_first  IF  A[i, j] > mean_a
       - A[i, j] * sum_b_second IF  A[i, j] <= mean_a
    """
    # --- Get program IDs to identify the current row ---
    # Each program instance processes one row of the matrices.
    pid_m = tl.program_id(axis=0)

    # --- Create pointers for the current row ---
    # Create a range of offsets for the columns [0, 1, 2, ..., BLOCK_SIZE_N-1]
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Pointers to the beginning of the current row for each matrix
    a_row_ptr = A_ptr + pid_m * stride_am
    b_row_ptr = B_ptr + pid_m * stride_bm
    c_row_ptr = C_ptr + pid_m * stride_cm
    
    # Pointers to all elements in the current row
    a_ptrs = a_row_ptr + offs_n
    b_ptrs = b_row_ptr + offs_n

    # --- Load the full rows of A and B into SRAM ---
    # Create a mask to avoid out-of-bounds access for N not divisible by BLOCK_SIZE_N
    mask_n = offs_n < N
    # Load data, using the mask. `other=0.0` ensures non-accessed elements are 0.
    row_a = tl.load(a_ptrs, mask=mask_n, other=0.0)
    row_b = tl.load(b_ptrs, mask=mask_n, other=0.0)

    # --- Perform calculations in SRAM ---
    # 1. Calculate the mean of the row from A
    mean_a = tl.sum(row_a, axis=0) / N

    # 2. & 3. Calculate sums of the two halves of the row from B
    N_HALF = N // 2
    # Create masks for the first and second halves of the row
    first_half_mask = offs_n < N_HALF
    
    # Apply masks to the loaded row_b to isolate values from each half
    b_first_half = tl.where(first_half_mask, row_b, 0.0)
    b_second_half = tl.where(~first_half_mask, row_b, 0.0) # ~ is logical NOT
    
    # Sum each half
    sum_b_first = tl.sum(b_first_half, axis=0)
    sum_b_second = tl.sum(b_second_half, axis=0)

    # 4. Apply the conditional logic
    # Create a boolean mask based on the condition
    condition_mask = row_a > mean_a
    
    # Use tl.where to select the appropriate sum for each element
    # This is equivalent to an element-wise if-else statement
    selected_sum = tl.where(condition_mask, sum_b_first, sum_b_second)
    
    # Calculate the final output value
    output = row_a * selected_sum

    # --- Store the result back to DRAM ---
    c_ptrs = c_row_ptr + offs_n
    tl.store(c_ptrs, output, mask=mask_n)


# ------------------------------------------------------------------
# 2. Python Wrapper for the Triton Kernel
# ------------------------------------------------------------------

def conditionally_weighted_aggregation_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper for launching the Triton kernel.
    """
    # Ensure inputs are on the GPU and have the same shape
    assert A.is_cuda and B.is_cuda, "Input tensors must be on a CUDA device"
    assert A.shape == B.shape, "Input matrices must have the same shape"
    
    M, N = A.shape
    
    # Create an empty output tensor
    C = torch.empty_like(A)
    
    # The grid defines how many instances of the kernel we launch.
    # Here, we launch M instances, one for each row.
    grid = (M, )
    
    # A compile-time constant for the kernel. Must be a power of two.
    # This helps Triton's compiler to unroll loops and optimize memory access.
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    # Launch the kernel
    conditionally_weighted_aggregation_kernel[grid](
        A, B, C,
        M, N,
        A.stride(0), B.stride(0), C.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return C

# ------------------------------------------------------------------
# 3. PyTorch Reference Implementation for Verification
# ------------------------------------------------------------------

def conditionally_weighted_aggregation_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A pure PyTorch implementation for verifying the correctness of the Triton kernel.
    """
    M, N = A.shape
    N_half = N // 2

    # 1. Calculate A's row means
    mean_a = A.mean(dim=1, keepdim=True)

    # 2. & 3. Calculate B's half-row sums
    sum_b_first = B[:, :N_half].sum(dim=1, keepdim=True)
    sum_b_second = B[:, N_half:].sum(dim=1, keepdim=True)

    # 4. Apply conditional logic using torch.where
    mask = A > mean_a
    C = torch.where(mask, A * sum_b_first, A * sum_b_second)

    return C

# ------------------------------------------------------------------
# 4. Main Block for Testing and Benchmarking
# ------------------------------------------------------------------

if __name__ == "__main__":
    # --- Setup Test Data ---
    torch.manual_seed(0)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define matrix dimensions for the test
    M, N = 1024, 2048
    
    # Create random input tensors on the GPU
    A = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    B = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    
    print(f"Testing with matrix shape: ({M}, {N}) on {DEVICE}")
    
    # --- Correctness Check ---
    print("Running correctness check...")
    output_triton = conditionally_weighted_aggregation_triton(A, B)
    output_pytorch = conditionally_weighted_aggregation_pytorch(A, B)
    
    # Compare the results. `atol` (absolute tolerance) and `rtol` (relative tolerance)
    # are used to account for minor floating-point differences.
    max_abs = torch.max(torch.abs(output_triton - output_pytorch)).item()
    print(f"Max absolute difference between Triton and PyTorch outputs: {max_abs:.6f}")
    