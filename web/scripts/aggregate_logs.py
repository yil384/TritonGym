#!/usr/bin/env python3
"""
Aggregate logs/<model>/<agent>/<operator>.json into web/frontend/public/data/leaderboard.json.

Usage:
    python web/scripts/aggregate_logs.py --logs logs --out web/frontend/public/data/leaderboard.json
"""
import argparse
import json
from pathlib import Path

DEFAULT_BENCHMARKS = [
    ("standard", "benchmark/data_v2/benchmark.json"),
    ("ood", "benchmark/data_ood_v2/benchmark.json"),
    ("dsl", "benchmark/data_dsl_v2/benchmark.json"),
]

# Display name + ordering for the leaderboard. Anything in logs/ not listed
# here falls through with a humanized display name and stable insertion order.
MODEL_DISPLAY = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-sonnet-4-20250514":   "Claude Sonnet 4",
    "claude-haiku-4-5-20251001":  "Claude Haiku 4.5",
    "gpt-5":                      "GPT-5",
    "gpt-4o":                     "GPT-4o",
    "deepseek-chat":              "DeepSeek-V3",
    "gemini-2.5-pro":             "Gemini 2.5 Pro",
    "gemini-1.5-pro":             "Gemini 1.5 Pro",
    "Qwen3-Coder-480B-A35B-Instruct-FP8": "Qwen-3",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4",
}

AGENT_DISPLAY = {
    "oneshot":     "One-shot",
    "geak":        "Geak",
    "alphaevolve": "AlphaEvolve",
    "leader":      "Leader",
}


def load_op_sets(base):
    sets = {}
    for name, rel in DEFAULT_BENCHMARKS:
        p = Path(base) / rel
        sets[name] = set(json.loads(p.read_text()).keys()) if p.exists() else set()
    return sets


def is_correct(r, abs_thresh=0.01):
    if not r.get("compile_success", False):
        return False
    m = r.get("metrics", {})
    if not isinstance(m, dict) or "error" in m:
        return False
    max_abs = m.get("max_abs")
    return max_abs is not None and max_abs <= abs_thresh


def perf_ratio(r):
    gen = r.get("runtime_ms")
    oracle = r.get("triton_runtime_ms")
    if not gen or gen <= 0 or not oracle:
        return None
    return oracle / gen


def metrics_for_split(results, op_set, abs_thresh=0.01):
    """Return (pass@1, perf@1, n_attempted, n_total) for the given split."""
    if not op_set:
        return None, None, 0, 0
    total = len(op_set)
    pass_count = 0
    perf_sum = 0.0
    attempted = 0
    for op in op_set:
        if op not in results:
            continue
        attempted += 1
        r = results[op]
        if is_correct(r, abs_thresh):
            pass_count += 1
            p = perf_ratio(r)
            if p is not None:
                perf_sum += p
    return pass_count / total, perf_sum / total, attempted, total


def _iter_model_agent_dirs(logs_path):
    """Yield (model_name, agent_dir) pairs.

    Supports two layouts:
      logs/<model>/<agent>/*.json
      logs/<provider>/<model>/<agent>/*.json   (e.g. Qwen, meta-llama)
    """
    KNOWN_AGENTS = {"oneshot", "geak", "alphaevolve", "leader"}
    for top in sorted(logs_path.iterdir()):
        if not top.is_dir():
            continue
        children = [c for c in top.iterdir() if c.is_dir()]
        if children and all(c.name in KNOWN_AGENTS for c in children):
            for c in sorted(children):
                yield top.name, c
        else:
            for sub in sorted(children):
                grandchildren = [g for g in sub.iterdir() if g.is_dir()]
                if grandchildren and all(g.name in KNOWN_AGENTS for g in grandchildren):
                    for g in sorted(grandchildren):
                        yield sub.name, g


def aggregate(logs_dir, base_dir):
    op_sets = load_op_sets(base_dir)
    all_ops = set().union(*op_sets.values())

    rows = []
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return rows, op_sets

    for model, agent_dir in _iter_model_agent_dirs(logs_path):
        agent = agent_dir.name
        results = {}
        tokens_in = 0
        tokens_out = 0
        for f in agent_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            results[f.stem] = data
            ti = data.get("total_input_tokens") or data.get("metadata", {}).get("total_input_tokens", 0) or 0
            to = data.get("total_output_tokens") or data.get("metadata", {}).get("total_output_tokens", 0) or 0
            tokens_in += ti
            tokens_out += to

        row = {
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "agent": agent,
            "agent_display": AGENT_DISPLAY.get(agent, agent.title()),
            "splits": {},
            "tokens": {"input": tokens_in, "output": tokens_out},
            "source": "logs",
        }
        for split, ops in op_sets.items():
            p, perf, attempted, total = metrics_for_split(results, ops)
            row["splits"][split] = {
                "pass": p, "perf": perf, "attempted": attempted, "total": total,
            }
        p, perf, attempted, total = metrics_for_split(results, all_ops)
        row["splits"]["full"] = {
            "pass": p, "perf": perf, "attempted": attempted, "total": total,
        }
        rows.append(row)
    return rows, op_sets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    p.add_argument("--base-dir", default=".")
    p.add_argument("--out", default="web/frontend/public/data/leaderboard.json")
    p.add_argument("--paper-results", default="web/scripts/paper_results.json",
                   help="Optional paper-reported numbers to merge in (preferred when present)")
    p.add_argument("--paper-only", action="store_true", default=True,
                   help="Emit only paper-reported rows (default). Use --include-logs to also include rows aggregated from logs/.")
    p.add_argument("--include-logs", dest="paper_only", action="store_false",
                   help="Also include rows aggregated from the local logs/ directory.")
    args = p.parse_args()

    rows, op_sets = aggregate(args.logs, args.base_dir)

    paper_path = Path(args.paper_results)
    paper_rows = []
    if paper_path.exists():
        paper_rows = json.loads(paper_path.read_text())

    # Index rows by (model_display, agent_display) so paper numbers win where they exist.
    by_key = {(r["model_display"], r["agent_display"]): r for r in rows}
    for pr in paper_rows:
        key = (pr["model_display"], pr["agent_display"])
        merged = by_key.get(key, {
            "model": pr.get("model", pr["model_display"]),
            "model_display": pr["model_display"],
            "agent": pr.get("agent", pr["agent_display"].lower()),
            "agent_display": pr["agent_display"],
            "tokens": pr.get("tokens", {"input": 0, "output": 0}),
        })
        # Paper numbers replace splits when provided.
        merged["splits"] = pr["splits"]
        merged["source"] = pr.get("source", "paper")
        if "tokens" in pr:
            merged["tokens"] = pr["tokens"]
        if "cost_usd" in pr:
            merged["cost_usd"] = pr["cost_usd"]
        by_key[key] = merged

    merged_rows = list(by_key.values())
    if args.paper_only:
        merged_rows = [r for r in merged_rows if r.get("source") == "paper"]

    out = {
        "splits": {
            name: {"size": len(ops)} for name, ops in op_sets.items()
        },
        "splits_order": ["standard", "ood", "dsl", "full"],
        "rows": merged_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote %d rows to %s" % (len(merged_rows), args.out))


if __name__ == "__main__":
    main()
