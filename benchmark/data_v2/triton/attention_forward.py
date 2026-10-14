import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, 
                    K_ptrs, K_scale_ptr, V_ptrs,  
                    stride_kn, stride_vk,
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    BLOCK_D: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    N_CTX: tl.constexpr):
    lo, hi = 0, N_CTX
    offs_d = tl.arange(0, BLOCK_D)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_valid = offs_n < (N_CTX - start_n)
        k_mask = n_valid[None, :] & (offs_d < HEAD_DIM)[:, None]
        k = tl.load(K_ptrs, mask = k_mask, other=0.0)
        k_scale = tl.load(K_scale_ptr)
        # q_scale/k_scale are 1; the log2(e) factor turns exp2 into the natural-exp softmax
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale * 1.44269504
        qk = tl.where(n_valid[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask = n_valid[:, None] & (offs_d < HEAD_DIM)[None, :], other=0.0)
        p = p.to(v.dtype)
        
        acc += tl.dot(p, v)  
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vk
    return acc, l_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,  
              stride_qz, stride_qh, stride_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              Z, H, N_CTX,  
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              BLOCK_D: tl.constexpr,
              STAGE: tl.constexpr  
              ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N) 
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D)
    d_mask = offs_k < HEAD_DIM
    Q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + k_offset + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    O_block_ptr = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask = (offs_m[:, None] < N_CTX) & d_mask[None, :], other=0.0)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,  
                                    stride_kn, stride_vk,
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N, BLOCK_D,  
                                    4 - STAGE, offs_m, offs_n, N_CTX 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < N_CTX) & d_mask[None, :])

def triton_attention_forward(q, k, v):
    q_scale = torch.ones((q.shape[0], q.shape[1], q.shape[2]), dtype=torch.float16, device=q.device)
    k_scale = torch.ones((k.shape[0], k.shape[1], k.shape[2]), dtype=torch.float16, device=k.device)
    BLOCK_M = 128
    BLOCK_N = 64
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert q.shape[:3] == k.shape[:3] == v.shape[:3]
    BLOCK_D = max(16, triton.next_power_of_2(HEAD_DIM_K))
    o = torch.empty_like(q, dtype=torch.float16)
    stage = 1

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,  
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  
        q.shape[0], q.shape[1],  
        N_CTX=q.shape[2],  
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K, BLOCK_D=BLOCK_D,  
        STAGE=stage,  
        num_warps=8,  
        num_stages=4)
    return o




##################################################################################################################################################


import torch

def test_forward():
    # Define the dimensions
    batch_size = 2  # Number of sequences
    num_heads = 4   # Number of attention heads
    seq_length = 128  # Length of each sequence
    head_dim = 128  # Dimension of each head

    # Create random input tensors
    q = torch.randn((batch_size, num_heads, seq_length, head_dim), dtype=torch.float16, device='cuda')
    k = torch.randn((batch_size, num_heads, seq_length, head_dim), dtype=torch.float16, device='cuda')
    v = torch.randn((batch_size, num_heads, seq_length, head_dim), dtype=torch.float16, device='cuda')

    # Create scale tensors
    q_scale = torch.ones((batch_size, num_heads, seq_length), dtype=torch.float32, device='cuda')
    k_scale = torch.ones((batch_size, num_heads, seq_length), dtype=torch.float32, device='cuda')

    # Call the forward function
    output = triton_attention_forward(q, k, v)

    # Store results in a dictionary
    results = {
        "test_case_1": output
    }
    return results

result_gold = test_forward()
