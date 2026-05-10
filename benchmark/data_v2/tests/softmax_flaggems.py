import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.softmax_flaggems import triton_softmax_flaggems
from benchmark.data_v2.torch.softmax_flaggems import torch_softmax_flaggems

def test_softmax():
    print("Testing softmax operator...")

    input_shapes = [
        ((10,),),           # 1D
        ((4, 5),),          # 2D
        ((4, 5),),          # 2D fp16
        ((2, 3, 4),),       # 3D
        ((2, 3, 4),),       # 3D fp64
        ((2, 3, 4, 1024),), # 4D large K
        ((1,),),            # single element
        ((1024, 1024),),    # large N
        ((2,),),            # Inf/-Inf
        ((1,),),            # NaN
        ((3, 7),),          # non-square
        ((2, 2),),          # small fp16
        ((512, 512),),      # large fp16
        ((2,),),            # extreme values
        ((2048, 2048),),    # very large
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    for idx, shape in enumerate(input_shapes):
        x = torch.randn(shape[0], device=device, dtype=dtype)
        
        # Triton output
        out_triton = triton_softmax_flaggems(x)

        # Torch output
        out_torch = torch_softmax_flaggems(x)

        # Compare
        max_abs = (out_triton - out_torch).abs().max().item()
        print(f"\033[92m OK \033[0m", end=' ')
        print(f"softmax case {idx}: shape={shape[0]}, dtype={dtype} max_abs={max_abs:.3e}")

if __name__ == "__main__":
    test_softmax()
