import torch

def _umulhi_32(a, b):
    """Upper 32 bits of unsigned 32x32->64 bit multiplication."""
    a64 = a.to(torch.int64) & 0xFFFFFFFF
    b64 = b.to(torch.int64) & 0xFFFFFFFF
    return ((a64 * b64) >> 32).to(torch.int32)

def _mul32(a, b):
    """Lower 32 bits of unsigned multiplication (wrapping)."""
    a64 = a.to(torch.int64) & 0xFFFFFFFF
    b64 = b.to(torch.int64) & 0xFFFFFFFF
    return (a64 * b64).to(torch.int32)

def _triton_rand(seed, offset):
    """
    Pure PyTorch implementation of Triton's tl.rand(seed, offset).
    Uses Philox-4x32-10 PRNG to match Triton's random number generation exactly.
    """
    device = offset.device

    PHILOX_KEY_A = torch.tensor(0x9E3779B9 - (1 << 32), dtype=torch.int32, device=device)
    PHILOX_KEY_B = torch.tensor(0xBB67AE85 - (1 << 32), dtype=torch.int32, device=device)
    ROUND_A = torch.tensor(0xD2511F53 - (1 << 32), dtype=torch.int32, device=device)
    ROUND_B = torch.tensor(0xCD9E8D57 - (1 << 32), dtype=torch.int32, device=device)

    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    k0 = torch.full_like(offset, seed & 0xFFFFFFFF, dtype=torch.int32)
    k1 = torch.full_like(offset, (seed >> 32) & 0xFFFFFFFF, dtype=torch.int32)

    c0 = offset.to(torch.int32)
    c1 = torch.zeros_like(c0, dtype=torch.int32)
    c2 = torch.zeros_like(c0, dtype=torch.int32)
    c3 = torch.zeros_like(c0, dtype=torch.int32)

    for _ in range(10):
        _c0, _c2 = c0.clone(), c2.clone()
        c0 = _umulhi_32(ROUND_B, _c2) ^ c1 ^ k0
        c2 = _umulhi_32(ROUND_A, _c0) ^ c3 ^ k1
        c1 = _mul32(ROUND_B, _c2)
        c3 = _mul32(ROUND_A, _c0)
        k0 = k0 + PHILOX_KEY_A
        k1 = k1 + PHILOX_KEY_B

    # uint_to_uniform_float
    x = c0
    x = torch.where(x < 0, -x - 1, x)
    scale = 4.6566127342e-10
    return x.float() * scale


def torch_seeded_dropout(x: torch.Tensor, p: float = 0.5, seed: int = 123) -> torch.Tensor:
    """
    Pure PyTorch implementation of Triton seeded dropout.
    Uses the same Philox PRNG as Triton's tl.rand to produce identical masks.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    n_elements = x.numel()
    offsets = torch.arange(n_elements, device=x.device, dtype=torch.int32)

    # Generate random numbers using Philox (matching tl.rand)
    random_numbers = _triton_rand(seed, offsets)

    # Apply dropout mask
    keep_mask = random_numbers > p
    output = torch.where(keep_mask, x.view(-1) / (1.0 - p), torch.zeros_like(x.view(-1)))

    return output.view_as(x)
