import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_dsl_v2.triton.gluon_memcpy_1d import triton_gluon_memcpy_1d
from benchmark.data_dsl_v2.torch.gluon_memcpy_1d import torch_gluon_memcpy_1d


if __name__ == "__main__":
    input_shapes =[
        ((1024,),),
        ((2048,),),
        ((4096,),),
        ((8192,),),
        ((16384,),),
        ((32768,),),
        ((65536,),),
        ((131072,),),
        ((262144,),),
        ((524288,),),
        ((1048576,),),
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for idx, shape in enumerate(input_shapes):
        input = torch.randn(shape[0], device=device, dtype=torch.float32)

        output = triton_gluon_memcpy_1d(input)
        output_ref = torch_gluon_memcpy_1d(input)
        try:
            max_abs = torch.max(torch.abs(output - output_ref)).item()
            print(f"[OK] Test {idx}: max absolute difference = {max_abs}")
        except Exception as e:
            print(f"[ERROR] Test {idx} failed: {e}")