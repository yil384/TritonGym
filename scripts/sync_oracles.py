#!/usr/bin/env python3
"""
Sync oracle triton implementations from triton/ files into benchmark.json.

Usage:
    python sync_oracles.py benchmark/data_v2/benchmark.json benchmark/data_v2/triton/
    python sync_oracles.py benchmark/data_ood_v2/benchmark.json benchmark/data_ood_v2/triton/
    python sync_oracles.py benchmark/data_dsl_v2/benchmark.json benchmark/data_dsl_v2/triton/

This updates the 'triton_operator' field in benchmark.json to match the
corresponding .py files in the triton/ directory.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: python sync_oracles.py <benchmark.json> <triton_dir/>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    triton_dir = Path(sys.argv[2])

    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    if not triton_dir.exists():
        print(f"ERROR: {triton_dir} not found")
        sys.exit(1)

    data = json.loads(json_path.read_text())
    updated = 0
    skipped = 0

    for op in sorted(data.keys()):
        triton_file = triton_dir / f"{op}.py"
        if not triton_file.exists():
            skipped += 1
            continue

        file_src = triton_file.read_text()
        old_src = data[op].get("triton_operator", "")

        if file_src.strip() != old_src.strip():
            data[op]["triton_operator"] = file_src
            updated += 1
            print(f"  UPDATED: {op}")
        else:
            skipped += 1

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nDone: {updated} updated, {skipped} unchanged. Written to {json_path}")


if __name__ == "__main__":
    main()
