import torch

def torch_spinning_lock_reduction(P: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of spinning lock reduction using meshgrid for safe indexing.
    """
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    M, N = P.shape
    num_sms = 5
    k = 3
    C = torch.zeros((M, N), device=P.device, dtype=P.dtype)
    locks = torch.zeros(num_sms, device=P.device, dtype=torch.int32)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    for pid in range(num_sms):
        pid_m = pid // num_sms
        pid_n = pid % num_sms

        acc = torch.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), device=P.device, dtype=P.dtype)

        # Perform reduction for every kth pid
        if pid % k == 0:
            next_pid = pid + 1
            while next_pid < pid + k and next_pid < num_sms:
                acc += P[next_pid*BLOCK_SIZE_M:(next_pid+1)*BLOCK_SIZE_M, :BLOCK_SIZE_N]
                next_pid += 1
            P[pid*BLOCK_SIZE_M:(pid+1)*BLOCK_SIZE_M, :BLOCK_SIZE_N] = acc
        else:
            acc += P[pid*BLOCK_SIZE_M:(pid+1)*BLOCK_SIZE_M, :BLOCK_SIZE_N]
            P[pid*BLOCK_SIZE_M:(pid+1)*BLOCK_SIZE_M, :BLOCK_SIZE_N] = acc

        # Compute row and column indices, respecting bounds
        rm = pid_m * BLOCK_SIZE_M + torch.arange(BLOCK_SIZE_M, device=P.device)
        rn = pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N, device=P.device)
        rm = rm[rm < M]
        rn = rn[rn < N]

        if rm.numel() > 0 and rn.numel() > 0:
            # Create meshgrid for 2D indexing
            rr, cc = torch.meshgrid(rm, rn, indexing='ij')
            C[rr, cc] = acc[:rm.numel(), :rn.numel()]

    return C