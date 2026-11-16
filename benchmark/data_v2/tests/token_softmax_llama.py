import sys

import torch

from benchmark.data_v2.triton.token_softmax_llama import triton_token_softmax_llama
from benchmark.data_v2.torch.token_softmax_llama import torch_token_softmax_llama
from benchmark.data_v2.tests._reference_check import (
    ReferenceReport,
    check_outputs,
    run_as_script,
    seed_case,
)

def test_token_softmax_llama():
    print("Testing token-wise softmax (LLAMA)...")

    # input_shapes: one (head_num, total_seq_len) logits tensor per case
    input_shapes = [
        ((2, 16),),
        ((5, 10),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # float32 is safer for softmax

    results = {}
    report = ReferenceReport("token_softmax_llama")

    for idx, shape in enumerate(input_shapes):

        # Generate random input from a fixed, logged seed (42 + case index)
        seed = seed_case(idx)
        Logics = torch.randn(shape[0], dtype=dtype, device=device)

        try:
            # Triton kernel
            triton_out = triton_token_softmax_llama(Logics)

            # Torch reference
            torch_out = torch_token_softmax_llama(Logics)

            # Shape, finiteness and maximum absolute error; raises on failure
            max_abs = check_outputs(torch_out, triton_out)
            report.passed(idx, [Logics], seed, max_abs)

            results[f"case_{idx}"] = torch_out

        except Exception as e:
            report.failed(idx, [Logics], seed, e)

    # Raises ReferenceTestFailed if any case failed
    report.finish()
    return results


if __name__ == "__main__":
    sys.exit(run_as_script(test_token_softmax_llama))
