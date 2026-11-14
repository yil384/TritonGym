#!/usr/bin/env python3
"""Run five real CUDA reference checks; missing CUDA is an error, never a skip."""

import argparse
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 0.01
SPECS = {
    "token_softmax_llama": ("float32", 1, [(2, 16), (5, 10)]),
    "attention_forward": ("float16", 3, [(2, 4, 128, 128), (2, 8, 256, 64), (1, 4, 512, 64)]),
}


def new_cases(operator):
    dtype, count, shapes = SPECS[operator]
    return [{"operator": operator, "case": i, "input_shapes": [list(shape)] * count,
             "dtype": dtype, "seed": 42 + i, "status": "not_run"}
            for i, shape in enumerate(shapes)]


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compare_outputs(expected, actual):
    """Reject broadcasting and non-finite values before computing absolute error."""
    import torch

    result = {"status": "failed", "max_abs_error": None, "tolerance": TOLERANCE}
    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        return {**result, "error": "Both outputs must be tensors"}
    result.update(expected_shape=list(expected.shape), actual_shape=list(actual.shape))
    if expected.shape != actual.shape:
        return {**result, "error": "Output shapes differ (broadcasting is forbidden)"}
    result.update(reference_finite=bool(torch.isfinite(expected).all().item()),
                  triton_finite=bool(torch.isfinite(actual).all().item()))
    if not result["reference_finite"] or not result["triton_finite"]:
        return {**result, "error": "Output contains NaN or Inf"}
    if expected.numel() == 0:
        return {**result, "error": "Empty outputs cannot establish correctness"}
    error = (expected.double() - actual.double()).abs().max().item()
    if not math.isfinite(error):
        return {**result, "error": "Absolute error is not finite"}
    result["max_abs_error"] = error
    return {**result, "status": "passed"} if error <= TOLERANCE else {
        **result, "error": "Maximum absolute error exceeds tolerance"}


def environment():
    info = {"python": platform.python_version(), "cuda": None, "cuda_available": False,
            "driver_version": None, "gpus": [], "device": "cuda:0", "git_sha": None, "git_dirty": None}
    for package in ("torch", "triton"):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info[package] = None
    if shutil.which("nvidia-smi"):
        try:
            lines = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, stderr=subprocess.PIPE, timeout=5).splitlines()
            info["driver_version"] = lines[0].strip() if lines else None
        except (OSError, subprocess.SubprocessError) as exc:
            info["driver_query_error"] = str(exc)
    try:
        info["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, timeout=10).strip())
    except (OSError, subprocess.SubprocessError) as exc:
        info["git_error"] = str(exc)
    try:
        import torch

        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    cuda_available=torch.cuda.is_available())
        if not info["cuda_available"] or info["cuda"] is None:
            return info, "CUDA is unavailable; a real CUDA-enabled PyTorch/GPU is required"
        info["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                         "capability": list(torch.cuda.get_device_capability(i))}
                        for i in range(torch.cuda.device_count())]
        importlib.import_module("triton")
    except Exception as exc:
        return info, f"GPU prerequisites unavailable: {type(exc).__name__}: {exc}"
    return info, None


def run_operator(operator, output):
    # Imports execute the original modules' CUDA self-tests in this child only.
    report = {"status": "failed", "cases": new_cases(operator)}
    write_report(output, report)
    try:
        import torch

        sys.path.insert(0, str(ROOT))
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        functions = [getattr(importlib.import_module(f"benchmark.data_v2.{kind}.{operator}"),
                             f"{kind}_{operator}") for kind in ("torch", "triton")]
        torch.cuda.synchronize()
        for case in report["cases"]:
            case["status"] = "running"
            write_report(output, report)
            torch.manual_seed(case["seed"])
            inputs = [torch.randn(shape, device="cuda:0", dtype=getattr(torch, case["dtype"]))
                      for shape in case["input_shapes"]]
            expected = functions[0](*(value.clone() for value in inputs))
            torch.cuda.synchronize()
            actual = functions[1](*(value.clone() for value in inputs))
            torch.cuda.synchronize()
            case.update(compare_outputs(expected, actual))
            write_report(output, report)
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        for case in report["cases"]:
            if case["status"] == "running":
                case.update(status="failed", error=report["error"])
    write_report(output, report)
    return 0 if report["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/reference-gpu.json"))
    parser.add_argument("--timeout", type=float, default=300, help="Timeout in seconds per operator")
    parser.add_argument("--operator", choices=SPECS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if args.operator:
        return run_operator(args.operator, args.output)
    info, error = environment()
    report = {"status": "unavailable" if error else "failed", "environment": info,
              "cases": [case for op in SPECS for case in new_cases(op)], "operator_errors": {}}
    if error:
        report["error"] = error
        write_report(args.output, report)
        print(f"unavailable: {error}; report: {args.output}", file=sys.stderr)
        return 2
    write_report(args.output, report)
    report["cases"] = []
    with tempfile.TemporaryDirectory(prefix="tritongym-gpu-") as directory:
        for operator in SPECS:
            output = Path(directory) / f"{operator}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--operator", operator,
                       "--output", str(output)]
            failure = None
            try:
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                           timeout=args.timeout, check=False)
                if completed.returncode:
                    failure = f"Worker exited {completed.returncode}: {completed.stderr[-4000:]}"
            except (OSError, subprocess.TimeoutExpired) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            try:
                worker = json.loads(output.read_text())
                if worker.get("error"):
                    failure = f"{failure}; {worker['error']}" if failure else worker["error"]
                if worker["status"] != "passed":
                    failure = failure or worker.get("error", "One or more cases failed or did not run")
            except (OSError, ValueError, KeyError) as exc:
                failure = failure or f"Missing or invalid worker report: {exc}"
                worker = {"cases": new_cases(operator)}
            if failure:
                report["operator_errors"][operator] = failure
                for case in worker["cases"]:
                    if case["status"] == "running":
                        case.update(status="failed", error=failure)
            report["cases"].extend(worker["cases"])
            write_report(args.output, report)
    report["status"] = "passed" if (not report["operator_errors"] and len(report["cases"]) == 5
                                    and all(c["status"] == "passed" for c in report["cases"])) else "failed"
    write_report(args.output, report)
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
