"""TritonGym leaderboard server.

A small FastAPI app that serves the static frontend, exposes the aggregated
leaderboard JSON, and the paper PDF. Mirrors picasso-lab's deployment shape
(uvicorn on 127.0.0.1:PORT behind nginx).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend" / "public"
DATA_FILE = FRONTEND_DIR / "data" / "leaderboard.json"

app = FastAPI(title="TritonGym Leaderboard")


@app.get("/api/leaderboard")
def leaderboard():
    if not DATA_FILE.exists():
        raise HTTPException(404, "leaderboard.json not found — run web/scripts/aggregate_logs.py")
    return JSONResponse(json.loads(DATA_FILE.read_text()))


@app.get("/api/health")
def health():
    return {"ok": True, "data_file_exists": DATA_FILE.exists()}


@app.get("/paper.pdf")
def paper():
    pdf = FRONTEND_DIR / "static" / "paper.pdf"
    if not pdf.exists():
        raise HTTPException(404, "paper.pdf not found")
    return FileResponse(pdf, media_type="application/pdf", filename="TritonGym.pdf")


# Mount the static frontend last so /api routes win.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7843"))
    uvicorn.run(app, host="127.0.0.1", port=port)
