"""CPU-only checks of the fail-closed GPU entrypoint, not GPU validation."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_reference_gpu.py"
SPEC = importlib.util.spec_from_file_location("check_reference_gpu", SCRIPT)
gpu_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_check)


def test_comparison_accepts_matching_finite_outputs():
    expected = torch.tensor([[0.25, 0.75], [0.1, 0.9]])
    result = gpu_check.compare_outputs(expected, expected.clone())
    assert result["status"] == "passed"
    assert result["max_abs_error"] == 0


def test_comparison_accepts_the_inclusive_absolute_tolerance():
    result = gpu_check.compare_outputs(torch.zeros(1, dtype=torch.float64),
                                       torch.tensor([0.01], dtype=torch.float64))
    assert result["status"] == "passed"
    assert result["max_abs_error"] == 0.01


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bad_reference", [False, True])
def test_comparison_rejects_non_finite_outputs(value, bad_reference):
    good, bad = torch.ones(2), torch.tensor([1.0, value])
    expected, actual = (bad, good) if bad_reference else (good, bad)
    result = gpu_check.compare_outputs(expected, actual)
    assert result["status"] == "failed"
    assert result["max_abs_error"] is None
    json.dumps(result, allow_nan=False)


def test_comparison_rejects_broadcastable_shape_mismatch():
    result = gpu_check.compare_outputs(torch.ones(2, 3), torch.ones(1, 3))
    assert result["status"] == "failed"
    assert "shapes differ" in result["error"]


def test_comparison_rejects_large_absolute_error():
    result = gpu_check.compare_outputs(torch.zeros(2), torch.tensor([0.0, 0.1]))
    assert result["status"] == "failed"
    assert result["max_abs_error"] > 0.01


def test_missing_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info, error = gpu_check.environment()
    assert info["cuda_available"] is False
    assert "CUDA is unavailable" in error


def test_cli_writes_five_not_run_cases_and_fails_without_cuda(tmp_path):
    output = tmp_path / "reports" / "reference-gpu.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "0"}
    completed = subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)],
                               cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert completed.returncode != 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "unavailable"
    assert len(report["cases"]) == 5
    assert all(case["status"] == "not_run" for case in report["cases"])
    assert [case["input_shapes"][0] for case in report["cases"]] == [
        [2, 16], [5, 10], [2, 4, 128, 128], [2, 8, 256, 64], [1, 4, 512, 64]]
    assert [case["dtype"] for case in report["cases"]] == ["float32"] * 2 + ["float16"] * 3
    assert report["environment"]["cuda_available"] is False
    assert all(key in report["environment"] for key in
               ("python", "torch", "triton", "cuda", "driver_version", "gpus", "git_sha", "git_dirty"))


@pytest.mark.parametrize("mode", ["failure", "timeout"])
def test_parent_failure_aggregation_control_only(monkeypatch, tmp_path, mode):
    """Exercise parent error handling on CPU; no GPU execution or success is simulated."""
    monkeypatch.setattr(gpu_check, "environment", lambda: ({"test_control_only": True}, None))

    def failing_worker(command, **kwargs):
        operator = command[command.index("--operator") + 1]
        output = Path(command[command.index("--output") + 1])
        cases = gpu_check.new_cases(operator)
        cases[0]["status"] = "running"
        output.write_text(json.dumps({"status": "failed", "cases": cases,
                                      "error": "synthetic worker failure"}))
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="synthetic process error")

    monkeypatch.setattr(gpu_check.subprocess, "run", failing_worker)
    output = tmp_path / "control-only.json"
    assert gpu_check.main(["--output", str(output), "--timeout", "1"]) == 1
    report = json.loads(output.read_text())
    assert report["environment"] == {"test_control_only": True}
    assert report["status"] == "failed"
    assert len(report["cases"]) == 5
    assert all(case["status"] in ("failed", "not_run") for case in report["cases"])
    assert sum(case["status"] == "failed" for case in report["cases"]) == 2
    assert len(report["operator_errors"]) == 2
    expected_error = "TimeoutExpired" if mode == "timeout" else "Worker exited 7"
    assert all(expected_error in error and "synthetic worker failure" in error
               for error in report["operator_errors"].values())
