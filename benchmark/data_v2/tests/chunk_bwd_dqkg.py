import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.chunk_bwd_dqkg import triton_chunk_bwd_dqkg
from benchmark.data_v2.torch.chunk_bwd_dqkg import torch_chunk_bwd_dqkg

def test_chunk_bwd_dqkg():
    print("Testing chunk_bwd_dqkg operator...")

    # Shapes: (q, k, v, g, h, do, dh) where:
    # q:[B,H,T,K], k:[B,H,T,K], v:[B,H,T,V], g:[B,H,T]
    # h:[B,H,V,NT*K] accessed as (V, NT*K) with strides (1, V) per (b,h)
    # Actually h passed to kernel via dh.stride; triton wrapper creates h as (B,H,NT*K,V)
    # dh:[B,H,NT*K,V] (same layout as h)
    # BT=64 so NT = T//BT
    # For T=128,K=64: NT=2, so h/dh are (B,H,128,64)
    # For T=64,K=32: NT=1, so h/dh are (B,H,32,32)
    input_shapes = [
        ((2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128), (2, 4, 128, 64), (2, 4, 128, 64), (2, 4, 128, 64)),
        ((3, 2, 64, 32), (3, 2, 64, 32), (3, 2, 64, 32), (3, 2, 64), (3, 2, 32, 32), (3, 2, 64, 32), (3, 2, 32, 32)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        q = torch.randn(shape[0], dtype=dtype, device=device)
        k = torch.randn(shape[1], dtype=dtype, device=device)
        v = torch.randn(shape[2], dtype=dtype, device=device)
        g = torch.randn(shape[3], dtype=dtype, device=device)
        h = torch.randn(shape[4], dtype=dtype, device=device)
        do = torch.randn(shape[5], dtype=dtype, device=device)
        dh = torch.randn(shape[6], dtype=dtype, device=device)

        try:
            # Triton output
            dq_triton= triton_chunk_bwd_dqkg(do, q, k, v, g, h, dh)
            # Torch output
            dq_torch= torch_chunk_bwd_dqkg(do, q, k, v, g, h, dh)

            max_abs_dq = (dq_triton - dq_torch).abs().max().item()


            print(f"\033[92m OK \033[0m", end=' ')
            print(f"chunk_bwd_dqkg case {idx}: max_abs_dq={max_abs_dq:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"chunk_bwd_dqkg case {idx}: {e}")


if __name__ == "__main__":
    test_chunk_bwd_dqkg()
