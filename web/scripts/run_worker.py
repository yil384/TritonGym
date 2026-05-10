#!/usr/bin/env python3
"""
Submission worker — picks pending HF-model submissions off the queue and
(eventually) runs them on a free GPU.

This is a SKELETON. The actual GPU sandbox + vLLM + benchmark.cli wiring is
deliberately not implemented here — running arbitrary user-submitted HF
models on a shared GPU server needs a sandbox we haven't built yet. Until
then this worker just *prints* what it would do and leaves submissions in
'pending' so a human can decide.

Real implementation (when sandbox is ready) replaces `_evaluate()` with:
    1. Download HF weights to /ssd2/yichen/tritongym-models/<sub.id>/
       (mind: must NOT go to /home/yichen — disk-budget rule)
    2. Pick a free GPU from {0,1,2,3} that's <30% utilized
       (mind: GPUs 4-7 off-limits + 70% utilization ceiling)
    3. Boot vLLM in a docker container with --gpus device=N restricted,
       fp16 weights, OAI-compatible endpoint at 127.0.0.1:<port>
    4. Run `python -m benchmark.cli --model http://127.0.0.1:<port>/v1
       --agent <agent> ...` for each requested agent, writing to
       logs/<hf_model_safe>/
    5. Run `python web/scripts/aggregate_logs.py --include-logs` to refresh
       the leaderboard
    6. Send a "your run completed" email via the Resend wiring used in
       backend/main.py (the same RESEND_API_KEY env var)
    7. Mark the submission completed (or failed, with stderr captured)

Usage:
    python web/scripts/run_worker.py                # one-shot, dry-run (default)
    python web/scripts/run_worker.py --loop         # poll every 60s
    python web/scripts/run_worker.py --commit       # actually run (NOT YET IMPLEMENTED)
"""
import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent.parent / "submissions.db"
POLL_INTERVAL_SECS = 60


def con():
    if not DB_FILE.exists():
        return None
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    return c


def _evaluate(sub: sqlite3.Row, commit: bool) -> tuple[bool, str]:
    """Run the evaluation for one submission.

    Returns (success, message). When `commit` is False (dry-run), only prints
    the steps and returns (False, 'dry-run') without touching the queue.

    NOT YET IMPLEMENTED for commit=True. The body sketches what each step
    would look like.
    """
    print(f"\n[#{sub['id']}] {sub['hf_model']} (agents={sub['agents']})")
    agents = sub["agents"].split(",")
    safe_name = sub["hf_model"].replace("/", "__")

    print("  step 1: download HF weights")
    print(f"          → /ssd2/yichen/tritongym-models/{safe_name}/")
    print("  step 2: pick a free GPU from {0,1,2,3} (<30% util, 70% ceiling)")
    print("  step 3: boot vLLM container with the picked GPU")
    print("  step 4: run benchmark.cli for each agent:")
    for a in agents:
        print(f"            python -m benchmark.cli --model <local-vllm> "
              f"--agent {a} --benchmark-json benchmark/data_v2/benchmark.json "
              f"--logs logs/{safe_name} --json")
    print("  step 5: aggregate_logs.py --include-logs")
    print(f"  step 6: notify {sub['email'] or '(no email on file)'}")
    print(f"  step 7: mark submission #{sub['id']} -> completed")

    if not commit:
        return False, "dry-run (rerun with --commit when sandbox is ready)"

    return False, "commit mode not implemented — sandbox + vLLM wiring needed"


def tick(commit: bool) -> int:
    """One pass over pending submissions. Returns number processed."""
    c = con()
    if c is None:
        print("submissions.db does not exist yet — queue is empty.")
        return 0
    rows = c.execute(
        "SELECT * FROM submissions WHERE status = 'pending' ORDER BY id ASC"
    ).fetchall()
    if not rows:
        print("queue is empty")
        return 0

    for sub in rows:
        ok, msg = _evaluate(sub, commit)
        if ok:
            c.execute("UPDATE submissions SET status = 'completed' WHERE id = ?", (sub["id"],))
            c.commit()
            print(f"[#{sub['id']}] completed — {msg}")
        else:
            print(f"[#{sub['id']}] skipped — {msg}")
    c.close()
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--loop", action="store_true",
                   help=f"poll continuously every {POLL_INTERVAL_SECS}s instead of one-shot")
    p.add_argument("--commit", action="store_true",
                   help="actually run evaluations (currently unimplemented; default is dry-run)")
    args = p.parse_args()

    if args.commit:
        print("WARNING: --commit specified, but the sandbox + vLLM wiring isn't built yet.\n"
              "         The worker will still skip every submission. Implement _evaluate() first.\n",
              file=sys.stderr)

    if args.loop:
        print(f"polling every {POLL_INTERVAL_SECS}s; Ctrl-C to stop")
        try:
            while True:
                tick(args.commit)
                time.sleep(POLL_INTERVAL_SECS)
        except KeyboardInterrupt:
            print("\nstopped")
    else:
        tick(args.commit)


if __name__ == "__main__":
    main()
