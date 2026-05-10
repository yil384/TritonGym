"""TritonGym leaderboard server.

A small FastAPI app that serves the static frontend, exposes the aggregated
leaderboard JSON, and the paper PDF. Mirrors picasso-lab's deployment shape
(uvicorn on 127.0.0.1:PORT behind nginx).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend" / "public"
DATA_FILE = FRONTEND_DIR / "data" / "leaderboard.json"
DB_FILE = ROOT / "submissions.db"

VALID_AGENTS = {"oneshot", "geak", "alphaevolve", "leader"}
HF_MODEL_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = FastAPI(title="TritonGym Leaderboard")


@contextmanager
def db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    try:
        con.execute(
            """CREATE TABLE IF NOT EXISTS submissions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                hf_model   TEXT    NOT NULL,
                email      TEXT,
                agents     TEXT    NOT NULL,
                notes      TEXT,
                ip         TEXT,
                status     TEXT    NOT NULL DEFAULT 'pending',
                created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        yield con
        con.commit()
    finally:
        con.close()


@app.get("/api/leaderboard")
def leaderboard():
    if not DATA_FILE.exists():
        raise HTTPException(404, "leaderboard.json not found — run web/scripts/aggregate_logs.py")
    return JSONResponse(json.loads(DATA_FILE.read_text()))


@app.get("/api/health")
def health():
    return {"ok": True, "data_file_exists": DATA_FILE.exists()}


@app.post("/api/submissions")
async def submit(request: Request):
    """Queue a HuggingFace model for evaluation.

    Submissions are written to a local SQLite file. A maintainer pulls them
    off the queue, runs the eval on a free GPU, then merges results into the
    leaderboard. No automatic execution — see web/scripts/run_submission.py.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "expected JSON body")

    hf = (payload.get("hf_model") or "").strip()
    email = (payload.get("email") or "").strip()
    agents = payload.get("agents") or []
    notes = (payload.get("notes") or "").strip()[:1000]

    if not HF_MODEL_RE.match(hf):
        raise HTTPException(400, "hf_model must look like 'org/name' (e.g. meta-llama/Llama-3.1-70B-Instruct)")
    if email and not EMAIL_RE.match(email):
        raise HTTPException(400, "email is invalid")
    if not isinstance(agents, list) or not agents or not all(a in VALID_AGENTS for a in agents):
        raise HTTPException(400, f"agents must be a non-empty subset of {sorted(VALID_AGENTS)}")

    ip = (request.headers.get("x-real-ip") or
          request.headers.get("x-forwarded-for", "").split(",")[0].strip() or
          (request.client.host if request.client else "unknown"))

    with db() as con:
        recent = con.execute(
            "SELECT COUNT(*) FROM submissions WHERE ip = ? AND created_at > datetime('now', '-1 day')",
            (ip,),
        ).fetchone()[0]
        if recent >= 5:
            raise HTTPException(429, "rate limit: max 5 submissions per IP per 24h")

        cur = con.execute(
            "INSERT INTO submissions(hf_model, email, agents, notes, ip) VALUES (?, ?, ?, ?, ?)",
            (hf, email or None, ",".join(agents), notes or None, ip),
        )
        sub_id = cur.lastrowid

    return {
        "id": sub_id,
        "status": "queued",
        "message": "Submission received. We'll run the eval when GPU time is available.",
    }


@app.get("/api/submissions/queue")
def queue_status():
    with db() as con:
        n_pending = con.execute(
            "SELECT COUNT(*) FROM submissions WHERE status = 'pending'"
        ).fetchone()[0]
        n_done = con.execute(
            "SELECT COUNT(*) FROM submissions WHERE status = 'completed'"
        ).fetchone()[0]
    return {"pending": n_pending, "completed": n_done}


@app.get("/paper.pdf")
def paper():
    pdf = FRONTEND_DIR / "static" / "paper.pdf"
    if not pdf.exists():
        raise HTTPException(404, "paper.pdf not found")
    return FileResponse(pdf, media_type="application/pdf", filename="TritonGym.pdf")


@app.middleware("http")
async def no_cache_html_js_css(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".html", ".js", ".css")) or path == "/" or path.endswith("/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# Mount the static frontend last so /api routes win.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7843"))
    uvicorn.run(app, host="127.0.0.1", port=port)
