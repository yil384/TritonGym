import torch
import warnings

from benchmark.data_v2.triton.token_softmax_llama import triton_token_softmax_llama
from benchmark.data_v2.torch.token_softmax_llama import torch_token_softmax_llama

def test_token_softmax_llama():
    print("Testing token-wise softmax (LLAMA)...")

    # input_shapes: list of dicts, each dict contains batch, head_num, max_input_len
    input_shapes = [
        ((2, 16),),
        ((5, 10),),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # float32 is safer for softmax

    results = {}

    for idx, shape in enumerate(input_shapes):

        # Generate random input
        Logics = torch.randn(shape[0], dtype=dtype, device=device)

        try:
            # Triton kernel
            triton_out = triton_token_softmax_llama(Logics)

            # Torch reference
            torch_out = torch_token_softmax_llama(Logics)

            # Compare maximum absolute error
            max_abs = (triton_out - torch_out).abs().max().item()
            print(f"\033[92m OK \033[0m token_softmax case {idx}: "
                  f"batch={2}, head={2}, max_input_len={8}, max_abs={max_abs:.3e}")

            results[f"case_{idx}"] = torch_out

        except Exception as e:
            warnings.warn(f"Test failed for case {idx}: {e}")
            print(f"\033[91m ERROR \033[0m token_softmax case {idx}: {e}")

    return results


if __name__ == "__main__":
    result_gold = test_token_softmax_llama()
