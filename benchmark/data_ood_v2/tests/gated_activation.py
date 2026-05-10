import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ------------------------------------------------------------------
# 1. Triton Kernel
# ------------------------------------------------------------------

@triton.jit
def fused_reduction_activation_kernel(
    # Pointers to tensors
    X_ptr,
    G_ptr,
    Y_ptr,
    # Tensor dimensions
    M, N,
    # Strides for memory access
    stride_xm, stride_xn,
    stride_gm,
    stride_ym, stride_yn,
    # Constants
    EPSILON: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for fused row-wise reduction and selective activation.
    
    For each row in X:
    1. Computes the max value and L1 norm of the row.
    2. Loads a corresponding gate value from G.
    3. Applies a fused transformation:
       Y = G * tanh(X / max) + (1 - G) * (X / (l1_norm + epsilon))
    """
    # Get the row index for this program instance
    pid = tl.program_id(axis=0)
    
    # --- Step 1: Compute row-wise max and L1 norm ---
    
    # Create pointers to the beginning of the current row in X and Y
    row_x_ptr = X_ptr + pid * stride_xm
    row_y_ptr = Y_ptr + pid * stride_ym
    
    # Create a range of offsets for the columns
    cols = tl.arange(0, BLOCK_SIZE_N)
    
    # Create a mask to handle rows that are not a multiple of BLOCK_SIZE_N
    mask = cols < N
    
    # Load the entire row into SRAM.
    # For rows larger than BLOCK_SIZE_N, this would need a loop.
    # Here, we assume N <= BLOCK_SIZE_N for simplicity, which is a common constraint.
    x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
    
    # Perform reductions in SRAM
    row_max = tl.max(x, axis=0)
    row_l1_norm = tl.sum(tl.abs(x), axis=0)
    
    # --- Step 2: Load the gate value ---
    
    # Pointer to the gate value for the current row
    gate_ptr = G_ptr + pid * stride_gm
    gate = tl.load(gate_ptr)
    
    # --- Step 3: Apply the transformation and store the result ---
    
    # The transformation is applied element-wise on the 'x' vector in SRAM
    term1 = gate * libdevice.tanh(x / row_max)
    term2 = (1 - gate) * (x / (row_l1_norm + EPSILON))
    y = term1 + term2
    
    # Write the final result from SRAM back to DRAM
    tl.store(row_y_ptr + cols * stride_yn, y, mask=mask)


# ------------------------------------------------------------------
# 2. Python Wrapper Function
# ------------------------------------------------------------------

def fused_reduction_activation(x: torch.Tensor, g: torch.Tensor, epsilon: float = 1e-5):
    """
    Python wrapper for launching the Triton kernel.
    
    Args:
        x (torch.Tensor): The input tensor of shape (M, N).
        g (torch.Tensor): The gate tensor of shape (M,).
        epsilon (float): A small value to avoid division by zero.
        
    Returns:
        torch.Tensor: The output tensor of shape (M, N).
    """
    M, N = x.shape
    
    # Ensure gate tensor has the correct shape
    assert g.shape == (M,), f"Gate tensor shape must be ({M},), but got {g.shape}"
    
    # Create the output tensor
    y = torch.empty_like(x)
    
    # The grid for launching the kernel is defined by the number of rows
    grid = (M,)
    
    # Find the next power of 2 for BLOCK_SIZE_N to optimize memory access
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    
    # Launch the kernel
    fused_reduction_activation_kernel[grid](
        x, g, y,
        M, N,
        x.stride(0), x.stride(1),
        g.stride(0),
        y.stride(0), y.stride(1),
        EPSILON=epsilon,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return y

# ------------------------------------------------------------------
# 3. PyTorch Reference Implementation for Verification
# ------------------------------------------------------------------

def reference_kernel_1(X, G, epsilon=1e-5):
    """
    A naive PyTorch implementation for correctness verification.
    """
    # Calculate per-row max value
    max_val = torch.max(X, dim=1, keepdim=True)[0]
    
    # Calculate per-row L1 norm
    l1_norm = torch.sum(torch.abs(X), dim=1, keepdim=True)
    
    # Unsqueeze the gate tensor to allow broadcasting
    gate = G.unsqueeze(1)
    
    # Apply the transformation
    term1 = gate * torch.tanh(X / max_val)
    term2 = (1 - gate) * (X / (l1_norm + epsilon))
    
    return term1 + term2

# ------------------------------------------------------------------
# 4. Main execution block for testing and benchmarking
# ------------------------------------------------------------------

if __name__ == "__main__":
    # --- Configuration ---
    M, N = 1024, 2048  # M: number of rows, N: number of columns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Create Input Tensors ---
    X = torch.randn((M, N), device=device, dtype=torch.float16)
    # Gate tensor with values between 0 and 1
    G = torch.rand(M, device=device, dtype=torch.float16)
    
    print(f"Running on device: {device}")
    print(f"Input tensor shape: {X.shape}")
    print("-" * 30)

    # --- Correctness Check ---
    print("Verifying correctness...")
    triton_output = fused_reduction_activation(X, G)
    pytorch_output = reference_kernel_1(X, G)
    
    # Using a slightly higher tolerance for float16
    is_correct = torch.allclose(triton_output, pytorch_output, atol=1e-2, rtol=1e-2)
    print(f"Outputs are close: {is_correct}")
    if not is_correct:
        print("Max difference:", (triton_output - pytorch_output).abs().max().item())
    print("-" * 30)

    # --- Performance Benchmark ---
    # The 'do_bench' function provides a convenient way to measure kernel performance
    print("Benchmarking performance...")
    triton_ms = triton.testing.do_bench(
        lambda: fused_reduction_activation(X, G),

    )
    pytorch_ms = triton.testing.do_bench(
        lambda: reference_kernel_1(X, G),
    )

    print(f"Speedup: {pytorch_ms / triton_ms:.2f}x")