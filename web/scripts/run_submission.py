#!/usr/bin/env python3
"""
Maintainer tool: list, claim, and resolve TritonGym HF-model submissions.

Submissions are queued by users via the website's POST /api/submissions and
land in web/submissions.db. There is no automatic worker — running an HF
model takes hours of GPU and you decide when to do it.

Typical flow:
    python web/scripts/run_submission.py list
    python web/scripts/run_submission.py show 7
    # then run the eval manually:
    #   python -m benchmark.cli --model meta-llama/... --agent oneshot ...
    # and once logs/<model>/ exists:
    python web/scripts/run_submission.py mark 7 completed
    python web/scripts/aggregate_logs.py --include-logs

Statuses: pending -> running -> completed   (or rejected)
"""
import argparse
import sqlite3
import sys
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent.parent / "submissions.db"


def con():
    if not DB_FILE.exists():
        print(f"no submissions.db yet at {DB_FILE} — has the website received any submissions?")
        sys.exit(0)
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    return c


def cmd_list(args):
    c = con()
    rows = c.execute(
        "SELECT id, hf_model, agents, status, created_at FROM submissions "
        "WHERE status = ? OR ? = 'all' ORDER BY id DESC",
        (args.status, args.status),
    ).fetchall()
    if not rows:
        print(f"no submissions with status={args.status}")
        return
    print("%-4s  %-50s  %-25s  %-10s  %s" % ("ID", "MODEL", "AGENTS", "STATUS", "CREATED"))
    print("-" * 110)
    for r in rows:
        print("%-4d  %-50s  %-25s  %-10s  %s" % (
            r["id"], r["hf_model"], r["agents"], r["status"], r["created_at"]
        ))


def cmd_show(args):
    c = con()
    r = c.execute("SELECT * FROM submissions WHERE id = ?", (args.id,)).fetchone()
    if not r:
        print(f"no submission with id={args.id}")
        sys.exit(1)
    for k in r.keys():
        print(f"{k:12s} : {r[k]}")


def cmd_mark(args):
    valid = {"pending", "running", "completed", "rejected"}
    if args.status not in valid:
        print(f"status must be one of {sorted(valid)}")
        sys.exit(1)
    c = con()
    cur = c.execute("UPDATE submissions SET status = ? WHERE id = ?", (args.status, args.id))
    c.commit()
    if cur.rowcount == 0:
        print(f"no submission with id={args.id}")
        sys.exit(1)
    print(f"submission #{args.id} -> {args.status}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list submissions")
    pl.add_argument("--status", default="pending", help="pending|running|completed|rejected|all")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="show one submission")
    ps.add_argument("id", type=int)
    ps.set_defaults(func=cmd_show)

    pm = sub.add_parser("mark", help="update submission status")
    pm.add_argument("id", type=int)
    pm.add_argument("status", help="pending|running|completed|rejected")
    pm.set_defaults(func=cmd_mark)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
