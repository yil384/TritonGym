import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.attention_forward import triton_attention_forward
from benchmark.data_v2.torch.attention_forward import torch_attention_forward
from benchmark.data_v2.tests._reference_check import (
    ReferenceReport,
    check_outputs,
    run_as_script,
    seed_case,
)


def test_attention_forward():
    print("Testing attention forward operator...")

    # Test case shape: (batch_size, num_heads, seq_length, head_dim)
    input_shapes = [
        ((2, 4, 128, 128), (2, 4, 128, 128), (2, 4, 128, 128)),
        ((2, 8, 256, 64), (2, 8, 256, 64), (2, 8, 256, 64)),
        ((1, 4, 512, 64), (1, 4, 512, 64), (1, 4, 512, 64)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = ReferenceReport("attention_forward")

    for idx, dim in enumerate(input_shapes):
        # Fixed, logged seed (42 + case index) so each case can be reproduced
        seed = seed_case(idx)

        # Triton kernel input must be float16
        q = torch.randn(dim[0], dtype=torch.float16, device=device)
        k = torch.randn(dim[1], dtype=torch.float16, device=device)
        v = torch.randn(dim[2], dtype=torch.float16, device=device)

        try:
            # Torch forward: softmax internally casts to float32, output cast back to float16
            torch_out = torch_attention_forward(q, k, v).to(torch.float16)

            # Triton forward: both input and output are float16
            triton_out = triton_attention_forward(q, k, v)

            # Shape, finiteness and maximum absolute error; raises on failure
            max_abs = check_outputs(torch_out, triton_out)
            report.passed(idx, [q, k, v], seed, max_abs)

        except Exception as e:
            report.failed(idx, [q, k, v], seed, e)

    # Raises ReferenceTestFailed if any case failed
    report.finish()


if __name__ == "__main__":
    sys.exit(run_as_script(test_attention_forward))
