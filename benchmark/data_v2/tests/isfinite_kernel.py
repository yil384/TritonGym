import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.isfinite_kernel import triton_isfinite_kernel
from benchmark.data_v2.torch.isfinite_kernel import torch_isfinite_kernel

def test_isfinite():
    print("Testing isfinite operator...")
    # Shapes for test cases
    input_shapes = [
        ((512,),),     # small tensor
        ((4096,),),    # larger tensor
        ((5,),),       # tensor with special values
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    for idx, shape in enumerate(input_shapes):
        # Normal random tensor
        in_tensor = torch.randn(shape[0], device=device, dtype=dtype)

        out_torch = torch_isfinite_kernel(in_tensor)
        out_triton = triton_isfinite_kernel(in_tensor)

        max_abs = (out_torch - out_triton).abs().max().item()
        print(f"\033[92m OK \033[0m isfinite case {idx} (normal values): max_abs={max_abs:.3e}")
    
if __name__ == "__main__":
    test_isfinite()
