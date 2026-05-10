#!/bin/bash
# deploy.sh — install the TritonGym leaderboard site on Ubuntu.
# Run from the web/ directory: bash deploy/deploy.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WEB_ROOT/.." && pwd)"

INSTALL_DIR="${INSTALL_DIR:-/opt/tritongym}"
SERVICE_USER="${SERVICE_USER:-${SUDO_USER:-$USER}}"
DOMAIN="${DOMAIN:-tritongym.picasso-lab.com}"
PORT="${PORT:-7843}"

echo "========================================"
echo "  TritonGym deploy"
echo "  install dir : $INSTALL_DIR"
echo "  run as user : $SERVICE_USER"
echo "  domain      : $DOMAIN"
echo "  port        : $PORT  (uvicorn, internal)"
echo "========================================"

# ── 1. System packages ──────────────────────────────────
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv nginx certbot python3-certbot-nginx curl

# ── 2. Refresh leaderboard.json from logs/ if present ──
echo ""
echo "[2/6] Aggregating logs into leaderboard.json (skipped if logs/ missing)..."
if [[ -d "$REPO_ROOT/logs" ]]; then
  python3 "$WEB_ROOT/scripts/aggregate_logs.py" \
    --logs "$REPO_ROOT/logs" \
    --base-dir "$REPO_ROOT" \
    --paper-results "$WEB_ROOT/scripts/paper_results.json" \
    --out "$WEB_ROOT/frontend/public/data/leaderboard.json"
else
  echo "    (no logs/ found — using existing leaderboard.json with paper numbers only)"
fi

# ── 3. Copy app files ───────────────────────────────────
echo ""
echo "[3/6] Copying app files to $INSTALL_DIR..."
sudo mkdir -p "$INSTALL_DIR"
sudo cp -r "$WEB_ROOT/backend"  "$INSTALL_DIR/"
sudo cp -r "$WEB_ROOT/frontend" "$INSTALL_DIR/"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# ── 4. Python venv ──────────────────────────────────────
echo "[4/6] Installing Python deps..."
cd "$INSTALL_DIR"
python3 -m venv venv
venv/bin/pip install -q -r backend/requirements.txt

# ── 5. systemd service ──────────────────────────────────
echo "[5/6] Configuring systemd service..."
sudo cp "$SCRIPT_DIR/tritongym.service" /etc/systemd/system/tritongym.service
sudo sed -i "s|User=ubuntu|User=$SERVICE_USER|g" /etc/systemd/system/tritongym.service
sudo sed -i "s|WorkingDirectory=.*|WorkingDirectory=$INSTALL_DIR/backend|g" /etc/systemd/system/tritongym.service
sudo sed -i "s|ExecStart=.*|ExecStart=$INSTALL_DIR/venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port $PORT|g" /etc/systemd/system/tritongym.service
sudo systemctl daemon-reload
sudo systemctl enable tritongym
sudo systemctl restart tritongym

# ── 6. nginx ────────────────────────────────────────────
echo "[6/6] Configuring nginx..."
sudo cp "$SCRIPT_DIR/nginx.conf.example" /etc/nginx/sites-available/tritongym
sudo sed -i "s|tritongym.picasso-lab.com|$DOMAIN|g; s|127.0.0.1:7843|127.0.0.1:$PORT|g" /etc/nginx/sites-available/tritongym
sudo ln -sf /etc/nginx/sites-available/tritongym /etc/nginx/sites-enabled/tritongym
sudo nginx -t && sudo systemctl reload nginx

# ── SSL ────────────────────────────────────────────────
echo ""
echo "DNS check: make sure $DOMAIN points to this server's public IP."
echo "Press Enter to request a Let's Encrypt cert (or Ctrl-C to skip)."
read -r
sudo certbot --nginx -d "$DOMAIN" \
  --non-interactive --agree-tos --email "admin@$DOMAIN" --redirect \
  || echo "[warn] certbot failed; rerun manually: sudo certbot --nginx -d $DOMAIN"

# ── Done ────────────────────────────────────────────────
echo ""
echo "Done."
echo "  service status : sudo systemctl status tritongym"
echo "  service logs   : sudo journalctl -u tritongym -f"
echo "  domain URL     : https://$DOMAIN"
