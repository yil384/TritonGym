#!/usr/bin/env python3
"""
Summarize benchmark results in the style of TritonGym Table 1.

Usage (from project root):
    python summarize_results.py --logs logs/deepseek-chat

Prints Pass@1 and Perf@1 broken down by Standard / OOD / DSL / Full.
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


# Must match the benchmark JSONs used by run_perlmutter.sh
DEFAULT_BENCHMARK_CONFIGS = [
    ("standard", "benchmark/data_v2/benchmark.json"),
    ("ood", "benchmark/data_ood_v2/benchmark.json"),
    ("dsl", "benchmark/data_dsl_v2/benchmark.json"),
]


def load_operator_sets(base_dir=".", configs=None):
    """Load operator names for each dataset category."""
    if configs is None:
        configs = DEFAULT_BENCHMARK_CONFIGS
    sets = {}
    for name, path in configs:
        p = Path(base_dir) / path
        if p.exists():
            sets[name] = set(json.loads(p.read_text()).keys())
        else:
            sets[name] = set()
    return sets


def load_results(log_dir):
    """Load all result JSONs from log_dir/{agent}/{operator}.json"""
    results = {}
    log_path = Path(log_dir)
    if not log_path.exists():
        return results
    for agent_dir in log_path.iterdir():
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name
        results[agent] = {}
        for f in agent_dir.glob("*.json"):
            op = f.stem
            try:
                data = json.loads(f.read_text())
                results[agent][op] = data
            except Exception:
                pass
    return results


def is_correct(result, abs_thresh=0.01, rel_thresh=None):
    """Check if a result passes correctness verification.

    Paper uses probabilistic verification with pre-defined thresholds.
    For fp16 kernels, only max_abs is checked (max_rel can be large due to
    small denominators in fp16).
    """
    if not result.get("compile_success", False):
        return False
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    if "error" in metrics:
        return False
    max_abs = metrics.get("max_abs")
    if max_abs is None:
        return False
    if max_abs > abs_thresh:
        return False
    if rel_thresh is not None:
        max_rel = metrics.get("max_rel")
        if max_rel is not None and max_rel > rel_thresh:
            return False
    return True


def compute_perf(result):
    """Compute Perf@1 = oracle_latency / generated_latency.

    Paper definition: Perf@K = max_{t: v_t=1} l^oracle / l_t
    Values > 1 mean generated code is faster than oracle.
    Values < 1 mean generated code is slower than oracle.
    """
    gen_ms = result.get("runtime_ms")
    oracle_ms = result.get("triton_runtime_ms")
    if gen_ms is None or oracle_ms is None:
        return None
    if gen_ms <= 0:
        return None
    return oracle_ms / gen_ms


def compute_metrics(results, op_set, abs_thresh=0.01, rel_thresh=None):
    """Compute Pass@1 and Perf@1 for a set of operators.

    Pass@1 = correct_count / total_operators (missing = failure)
    Perf@1 = average of (oracle_ms / gen_ms) over ALL operators
             (0 for failed operators, per paper definition)
    """
    if not op_set:
        return None, None

    pass_count = 0
    perf_sum = 0.0
    total = len(op_set)

    for op in op_set:
        if op not in results:
            # Missing = failure, perf = 0
            continue
        r = results[op]
        if is_correct(r, abs_thresh, rel_thresh):
            pass_count += 1
            perf = compute_perf(r)
            if perf is not None:
                perf_sum += perf

    pass_rate = pass_count / total if total > 0 else 0
    avg_perf = perf_sum / total if total > 0 else 0
    return pass_rate, avg_perf


def fmt_pct(val):
    if val is None:
        return "  N/A"
    return "%5.1f%%" % (val * 100)


def fmt_perf(val):
    if val is None:
        return " N/A"
    return "%6.3f" % val


def print_table(results, op_sets, abs_thresh, rel_thresh):
    """Print results table like Table 1 in the paper."""
    agents = sorted(results.keys())

    W = 106
    print("\n" + "=" * W)
    print("  Evaluation Results (TritonGym Table 1 Style)")
    print("=" * W)
    print("%-20s | %20s | %20s | %20s | %20s" % ("Agent", "Standard Samples", "OOD Samples", "DSL Samples", "Full Benchmark"))
    print("%-20s | %9s %10s | %9s %10s | %9s %10s | %9s %10s" % ("", "Pass@1", "Perf@1", "Pass@1", "Perf@1", "Pass@1", "Perf@1", "Pass@1", "Perf@1"))
    print("-" * W)

    for agent in agents:
        agent_results = results[agent]

        std_pass, std_perf = compute_metrics(agent_results, op_sets.get("standard", set()), abs_thresh, rel_thresh)
        ood_pass, ood_perf = compute_metrics(agent_results, op_sets.get("ood", set()), abs_thresh, rel_thresh)
        dsl_pass, dsl_perf = compute_metrics(agent_results, op_sets.get("dsl", set()), abs_thresh, rel_thresh)

        all_ops = set()
        for s in op_sets.values():
            all_ops |= s
        full_pass, full_perf = compute_metrics(agent_results, all_ops, abs_thresh, rel_thresh)

        print("%-20s | %9s %10s | %9s %10s | %9s %10s | %9s %10s" % (
            agent,
            fmt_pct(std_pass), fmt_perf(std_perf),
            fmt_pct(ood_pass), fmt_perf(ood_perf),
            fmt_pct(dsl_pass), fmt_perf(dsl_perf),
            fmt_pct(full_pass), fmt_perf(full_perf),
        ))

    print("-" * W)

    # Dataset sizes
    std_n = len(op_sets.get("standard", set()))
    ood_n = len(op_sets.get("ood", set()))
    dsl_n = len(op_sets.get("dsl", set()))
    print("\nDataset sizes: Standard=%d, OOD=%d, DSL=%d" % (std_n, ood_n, dsl_n))

    # Completion stats per dataset
    print("\nCompletion status:")
    all_ops = set()
    for s in op_sets.values():
        all_ops |= s
    for agent in agents:
        n = len(results[agent])
        parts = []
        for cat in ["standard", "ood", "dsl"]:
            cat_ops = op_sets.get(cat, set())
            cat_done = len(cat_ops & set(results[agent].keys()))
            parts.append("%s=%d/%d" % (cat, cat_done, len(cat_ops)))
        print("  %s: %d results collected (total operators: %d) [%s]" % (
            agent, n, len(all_ops), ", ".join(parts)))

    # Token usage
    print("\nToken usage:")
    for agent in agents:
        total_in = 0
        total_out = 0
        for r in results[agent].values():
            ti = r.get("total_input_tokens", 0)
            if not ti:
                ti = r.get("metadata", {}).get("total_input_tokens", 0)
            to = r.get("total_output_tokens", 0)
            if not to:
                to = r.get("metadata", {}).get("total_output_tokens", 0)
            total_in += ti or 0
            total_out += to or 0
        if total_in > 0 or total_out > 0:
            print("  %s: input=%s tokens, output=%s tokens, total=%s" % (
                agent, format(total_in, ","), format(total_out, ","), format(total_in + total_out, ",")))
        else:
            print("  %s: (no token data - run with latest code to track)" % agent)

    # Failure breakdown
    print("\nFailure breakdown:")
    for agent in agents:
        compile_fail = 0
        runtime_err = 0
        verify_fail = 0
        correct = 0
        total = 0
        for r in results[agent].values():
            total += 1
            if not r.get("compile_success", False):
                compile_fail += 1
            elif "error" in r.get("metrics", {}):
                runtime_err += 1
            elif not is_correct(r, abs_thresh, rel_thresh):
                verify_fail += 1
            else:
                correct += 1
        if total > 0:
            print("  %s: Correct=%d(%.0f%%) | CompileErr=%d(%.0f%%) | RuntimeErr=%d(%.0f%%) | VerifyFail=%d(%.0f%%)" % (
                agent,
                correct, correct / total * 100,
                compile_fail, compile_fail / total * 100,
                runtime_err, runtime_err / total * 100,
                verify_fail, verify_fail / total * 100,
            ))


def main():
    p = argparse.ArgumentParser(description="Summarize TritonGym benchmark results")
    p.add_argument("--logs", default="logs/deepseek-chat", help="Path to model log directory")
    p.add_argument("--base-dir", default=".", help="Project base directory")
    p.add_argument("--abs-thresh", type=float, default=0.01,
                   help="Max absolute error threshold for Pass@1 (default: 0.01)")
    p.add_argument("--rel-thresh", type=float, default=None,
                   help="Max relative error threshold for Pass@1 (default: None, only check abs)")
    args = p.parse_args()

    op_sets = load_operator_sets(args.base_dir)
    results = load_results(args.logs)

    if not results:
        print("No results found in %s" % args.logs)
        return

    print("\nModel: %s" % Path(args.logs).name)
    if args.rel_thresh is not None:
        print("Correctness thresholds: max_abs <= %s, max_rel <= %s" % (args.abs_thresh, args.rel_thresh))
    else:
        print("Correctness thresholds: max_abs <= %s (max_rel not checked)" % args.abs_thresh)
    print_table(results, op_sets, args.abs_thresh, args.rel_thresh)


if __name__ == "__main__":
    main()
