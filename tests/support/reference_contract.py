"""Exercise legacy test scripts with CPU stand-ins, never real Triton kernels.

These checks test a script's reporting/exit contract, not GPU correctness.
Each invocation runs in a fresh process so module substitutions cannot leak.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import json
from pathlib import Path
import runpy
import subprocess
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CASE_COUNTS = {"token_softmax_llama": 2, "attention_forward": 3}
SCENARIOS = ("correct", "bad_value", "nan", "inf", "wrong_shape", "runtime_error")
CALL_PREFIX = "REFERENCE_CONTRACT_CALL "
COMPLETE_PREFIX = "REFERENCE_CONTRACT_COMPLETE "


def run_reference_script(
    operator: str, scenario: str = "correct", *, timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    """Run one script/scenario; preserve its exit status and captured output."""
    if operator not in EXPECTED_CASE_COUNTS:
        raise ValueError(f"Unknown operator: {operator!r}")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    return subprocess.run(
        [sys.executable, "-m", "tests.support.reference_contract", operator, scenario],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def call_trace(result: subprocess.CompletedProcess[str]) -> list[dict]:
    return [
        json.loads(line[len(CALL_PREFIX):])
        for line in result.stdout.splitlines()
        if line.startswith(CALL_PREFIX)
    ]


def describe_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"exit code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


class _BlockRealTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "triton"
            or fullname.startswith("triton.")
            or fullname.startswith("benchmark.data_v2.triton.")
        ):
            raise ImportError(f"Real Triton imports are forbidden in CPU checks: {fullname}")
        return None


def _run_child(operator: str, scenario: str) -> None:
    sys.meta_path.insert(0, _BlockRealTriton())
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(42)
    # The same command must stay on CPU even when launched on a GPU machine.
    torch.cuda.is_available = lambda: False
    reference_module = importlib.import_module(f"benchmark.data_v2.torch.{operator}")
    reference = getattr(reference_module, f"torch_{operator}")
    calls = 0
    successful_returns = 0

    def stand_in(*args, **kwargs):
        nonlocal calls, successful_returns
        calls += 1
        inputs = [
            {"shape": list(arg.shape), "dtype": str(arg.dtype), "device": str(arg.device)}
            for arg in args
        ]
        print(CALL_PREFIX + json.dumps({
            "operator": operator, "scenario": scenario, "case": calls, "inputs": inputs,
        }), flush=True)
        if any(arg.device.type != "cpu" for arg in args):
            raise RuntimeError("CPU contract check received a non-CPU input")
        if scenario == "runtime_error":
            raise RuntimeError("Injected kernel runtime failure")
        out = reference(*args, **kwargs)
        if scenario == "bad_value":
            out = out + 1
        elif scenario == "nan":
            out = torch.full_like(out, float("nan"))
        elif scenario == "inf":
            out = torch.full_like(out, float("inf"))
        elif scenario == "wrong_shape":
            # Broadcasting would preserve every value, so exact shape matters.
            out = out.unsqueeze(0)
        successful_returns += 1
        return out

    module_name = f"benchmark.data_v2.triton.{operator}"
    replacement = types.ModuleType(module_name)
    setattr(replacement, f"triton_{operator}", stand_in)
    sys.modules[module_name] = replacement
    try:
        runpy.run_module(f"benchmark.data_v2.tests.{operator}", run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    if scenario == "correct":
        expected = EXPECTED_CASE_COUNTS[operator]
        if calls != expected or successful_returns != expected:
            raise RuntimeError(
                f"Incomplete positive control: {calls} calls, "
                f"{successful_returns} returns; expected {expected} of each"
            )
        print(COMPLETE_PREFIX + json.dumps({
            "operator": operator, "calls": calls, "successful_returns": successful_returns,
        }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", choices=EXPECTED_CASE_COUNTS)
    parser.add_argument("scenario", choices=SCENARIOS)
    args = parser.parse_args()
    _run_child(args.operator, args.scenario)
