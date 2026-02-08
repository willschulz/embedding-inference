#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
UNIT_SRC="$REPO_DIR/systemd/embedder.service"
UNIT_DST="/etc/systemd/system/embedder.service"

echo "==> Installing embedder.service …"
sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload

echo "==> Enabling embedder.service on boot …"
sudo systemctl enable embedder.service

echo "==> Starting embedder.service …"
sudo systemctl start embedder.service

echo ""
echo "✓ Done.  Check status with:"
echo "    sudo systemctl status embedder.service"
echo "    sudo journalctl -u embedder -f"
