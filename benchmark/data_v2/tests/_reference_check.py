"""Shared output checks and reporting for the reference test scripts.

A reference test compares a Triton kernel with its Torch reference. This
module makes a wrong result visible in the process exit status: a script
that uses it exits zero only when every case passed every check.

Each case is seeded with ``BASE_SEED + case index`` before its inputs are
created, so any single case can be reproduced on its own. This is the same
seeding used by ``scripts/check_reference_gpu.py``.
"""

import sys
import traceback

import torch

TOLERANCE = 0.01  # maximum allowed absolute error (the benchmark's Pass@1 limit)
BASE_SEED = 42

_GREEN, _RED, _RESET = "\033[92m", "\033[91m", "\033[0m"


class OutputCheckError(AssertionError):
    """The Triton output failed a shape, finiteness or tolerance check."""


class ReferenceTestFailed(AssertionError):
    """At least one case of a reference test failed."""


def seed_case(case):
    """Seed Torch for one case and return the seed that was used."""
    seed = BASE_SEED + case
    torch.manual_seed(seed)
    return seed


def check_outputs(torch_out, triton_out, tolerance=TOLERANCE):
    """Return the maximum absolute error, or raise OutputCheckError.

    The shape check comes before the subtraction so that broadcasting cannot
    hide a wrong output shape. NaN and infinity are failures in either output.
    """
    for name, out in (("Torch", torch_out), ("Triton", triton_out)):
        if not isinstance(out, torch.Tensor):
            raise OutputCheckError(f"{name} output is {type(out).__name__}, not a tensor")
    if torch_out.shape != triton_out.shape:
        raise OutputCheckError(
            f"output shapes differ: Torch {tuple(torch_out.shape)} "
            f"vs Triton {tuple(triton_out.shape)}"
        )
    for name, out in (("Torch", torch_out), ("Triton", triton_out)):
        bad = out.numel() - int(torch.isfinite(out).sum().item())
        if bad:
            raise OutputCheckError(f"{name} output has {bad} non-finite value(s) (NaN or Inf)")
    max_abs = (torch_out.double() - triton_out.double()).abs().max().item()
    if not max_abs <= tolerance:  # written this way so a NaN error also fails
        raise OutputCheckError(f"max_abs={max_abs:.3e} exceeds the limit {tolerance:.1e}")
    return max_abs


class ReferenceReport:
    """Log one PASS/FAIL line per case and remember the failures."""

    def __init__(self, operator):
        self.operator = operator
        self.total = 0
        self.failures = []

    def _context(self, case, inputs, seed):
        shapes = ", ".join(str(tuple(x.shape)) for x in inputs)
        return (
            f"{self.operator} case {case}: input_shapes=[{shapes}] "
            f"dtype={inputs[0].dtype} device={inputs[0].device} seed={seed}"
        )

    def passed(self, case, inputs, seed, max_abs):
        self.total += 1
        print(f"{_GREEN} PASS {_RESET} {self._context(case, inputs, seed)} "
              f"max_abs={max_abs:.3e} (limit {TOLERANCE:.1e})", flush=True)

    def failed(self, case, inputs, seed, error):
        """Record a failed case. Call this from inside the ``except`` block."""
        self.total += 1
        if isinstance(error, OutputCheckError):
            reason = str(error)
        else:
            reason = f"{type(error).__name__}: {error}"
            traceback.print_exc()  # keep the full runtime error for reproduction
        self.failures.append((case, reason))
        print(f"{_RED} FAIL {_RESET} {self._context(case, inputs, seed)} "
              f"reason={reason}", flush=True)

    def finish(self):
        """Print the summary; raise ReferenceTestFailed if any case failed."""
        passed = self.total - len(self.failures)
        print(f"{self.operator}: {passed}/{self.total} cases passed", flush=True)
        if self.failures:
            cases = ", ".join(str(case) for case, _ in self.failures)
            raise ReferenceTestFailed(
                f"{self.operator}: {len(self.failures)} of {self.total} cases failed "
                f"(case {cases})"
            )


def run_as_script(test_function):
    """Run a reference test and return its process exit status (0 or 1)."""
    try:
        test_function()
    except ReferenceTestFailed as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    return 0
