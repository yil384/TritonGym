"""Shared output comparison for the benchmark evaluator.

Used by both the agent-facing tools (``benchmark/eval/tools.py``) and the scoring command
(``benchmark/cli.py``) so that feedback and final scores apply the same rule.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

TOLERANCE = 1e-2


def _mismatch(reason: str) -> Dict[str, Any]:
    return {"max_abs": float("inf"), "max_rel": float("inf"), "mismatch": reason}


def compare_outputs(out: Any, ref: torch.Tensor) -> Dict[str, Any]:
    """Compare one generated output with one reference output.

    Returns ``max_abs`` and ``max_rel``. When the output cannot be a correct answer whatever
    the tolerance (not a tensor, a different shape, or NaN/infinity where the reference has
    none), both are infinity and ``mismatch`` explains why.
    """
    if not torch.is_tensor(out):
        return _mismatch(f"expected a tensor, got {type(out).__name__}")
    # Subtraction broadcasts, so shapes must be compared before any arithmetic.
    if tuple(out.shape) != tuple(ref.shape):
        return _mismatch(f"output shape {tuple(out.shape)} differs from reference shape {tuple(ref.shape)}")
    with torch.no_grad():
        out64 = out.to(torch.float64)
        ref64 = ref.to(torch.float64)
        ref_finite = torch.isfinite(ref64)
        # A non-finite value is only acceptable where the reference has the same one.
        same_nonfinite = (torch.isnan(out64) & torch.isnan(ref64)) | (out64 == ref64)
        bad = (torch.isfinite(out64) != ref_finite) | (~ref_finite & ~same_nonfinite)
        if bool(bad.any()):
            return _mismatch(
                f"{int(bad.sum())} of {bad.numel()} values are NaN or infinity where the reference "
                "is not, or the reverse"
            )
        if not bool(ref_finite.any()):
            return {"max_abs": 0.0, "max_rel": 0.0}
        diff = (out64 - ref64).abs()[ref_finite]
        rel = diff / ref64[ref_finite].abs().clamp_min(1e-8)
        return {"max_abs": diff.max().item(), "max_rel": rel.max().item()}


def mismatch_reason(metrics: Dict[str, Any]) -> str:
    """Return the first ``mismatch`` reason in single- or multi-output metrics, or ''."""
    if metrics.get("mismatch"):
        return str(metrics["mismatch"])
    for index, per_output in enumerate(metrics.get("per_output") or []):
        if isinstance(per_output, dict) and per_output.get("mismatch"):
            return f"output {index}: {per_output['mismatch']}"
    return ""
