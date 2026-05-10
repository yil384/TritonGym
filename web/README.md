# TritonGym Leaderboard Site

Static-ish leaderboard site (FastAPI backend + vanilla HTML/CSS/JS frontend),
designed to be deployed at `tritongym.picasso-lab.com`.

## Layout

```
web/
├── backend/
│   ├── main.py            # FastAPI: serves /api/leaderboard, paper.pdf, static files
│   └── requirements.txt
├── frontend/public/       # everything below is served as static files
│   ├── index.html         # leaderboard SPA
│   ├── style.css
│   ├── app.js
│   ├── data/leaderboard.json   # generated; checked in for static-mode fallback
│   └── static/             # paper.pdf + figures (overview.png, dataset.png, …)
├── scripts/
│   ├── aggregate_logs.py  # walk logs/<model>/<agent>/*.json -> leaderboard.json
│   └── paper_results.json # paper-reported numbers, merged in by aggregate_logs.py
└── deploy/
    ├── deploy.sh
    ├── nginx.conf.example
    └── tritongym.service
```

## Local dev

```bash
# refresh leaderboard.json from logs/ and paper_results.json
python3 web/scripts/aggregate_logs.py

# start the dev server
pip install -r web/backend/requirements.txt
PORT=7843 python3 web/backend/main.py
# open http://127.0.0.1:7843
```

## Deploy to tritongym.picasso-lab.com

Mirrors the picasso-lab pattern (uvicorn on 127.0.0.1 behind nginx, certbot for HTTPS).

```bash
cd web
DOMAIN=tritongym.picasso-lab.com bash deploy/deploy.sh
```

The script: installs system packages, regenerates `leaderboard.json` from `logs/`,
copies `web/{backend,frontend}` to `/opt/tritongym`, installs the systemd unit
on port 7843, drops the nginx vhost, then runs certbot.

## Updating the leaderboard

1. Drop new logs into `logs/<model>/<agent>/<operator>.json`
2. `python3 web/scripts/aggregate_logs.py` to regenerate `leaderboard.json`
3. `sudo systemctl restart tritongym` (or just refresh — the JSON file is read on every request)

To override a row with paper-reported numbers, edit `web/scripts/paper_results.json`.
