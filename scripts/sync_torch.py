#!/usr/bin/env python3
"""
Sync torch reference implementations from torch/ files into benchmark.json.

Usage:
    python sync_torch.py benchmark/data_v2/benchmark.json benchmark/data_v2/torch/
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: python sync_torch.py <benchmark.json> <torch_dir/>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    torch_dir = Path(sys.argv[2])

    data = json.loads(json_path.read_text())
    updated = 0

    for op in sorted(data.keys()):
        torch_file = torch_dir / f"{op}.py"
        if not torch_file.exists():
            continue

        file_src = torch_file.read_text()
        old_src = data[op].get("torch_operator", "")

        if file_src.strip() != old_src.strip():
            data[op]["torch_operator"] = file_src
            updated += 1
            print(f"  UPDATED: {op}")

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nDone: {updated} torch operators synced to {json_path}")


if __name__ == "__main__":
    main()
