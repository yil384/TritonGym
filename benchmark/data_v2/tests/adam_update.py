import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from benchmark.data_v2.triton.adam_update import triton_adam_update
from benchmark.data_v2.torch.adam_update import torch_adam_update

def test_adam_update():
    print("Testing Adam update operator...")

    # Shapes for (parameter, grad, exp_avg)
    input_shapes = [
        ((128,), (128,), (128,)),             # case 1: 1D vector
        ((1024,), (1024,), (1024,)),          # case 2: large 1D vector
        ((32, 32), (32, 32), (32, 32)),       # case 3: 2D matrix
        ((64, 128), (64, 128), (64, 128)),    # case 4: larger matrix
        ((4, 8, 16), (4, 8, 16), (4, 8, 16)), # case 5: 3D tensor
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Adam is generally more stable with float32

    def parse_triplet(entry):
        if (
            isinstance(entry, (list, tuple)) 
            and len(entry) == 3 
            and all(isinstance(x, (list, tuple)) for x in entry)
        ):
            return tuple(entry[0]), tuple(entry[1]), tuple(entry[2])
        raise ValueError(f"Invalid input shape entry: {entry}")

    # Hyperparameters
    lr = 0.01
    wd = 0.01
    beta1 = 0.9
    beta2 = 0.999

    for idx, entry in enumerate(input_shapes):
        Pshape, Gshape, Eshape = parse_triplet(entry)

        p = torch.randn(*Pshape, device=device, dtype=dtype)
        grad = torch.randn(*Gshape, device=device, dtype=dtype)
        exp_avg = torch.zeros(*Eshape, device=device, dtype=dtype)

        p_torch = p.clone()
        exp_avg_torch = exp_avg.clone()
        p_triton = p.clone()
        exp_avg_triton = exp_avg.clone()

        try:
            torch_adam_update(p_torch, grad, exp_avg_torch, lr, wd, beta1, beta2)
            triton_adam_update(p_triton, grad, exp_avg_triton, lr, wd, beta1, beta2)

            max_abs_p = (p_torch - p_triton).abs().max().item()
            max_abs_exp = (exp_avg_torch - exp_avg_triton).abs().max().item()

            print(f"\033[92m OK \033[0m", end=' ')
            print(f"Adam case {idx}: P{Pshape} G{Gshape} E{Eshape} "
                  f"max_abs_p={max_abs_p:.3e} max_abs_exp={max_abs_exp:.3e}")
        except Exception as e:
            print(f"\033[91m ERROR \033[0m", end=' ')
            print(f"Adam case {idx}: P{Pshape} G{Gshape} E{Eshape}: {e}")


if __name__ == "__main__":
    test_adam_update()
