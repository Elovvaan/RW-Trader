#!/usr/bin/env bash
# ============================================================
# deploy-local.sh — run this on YOUR LOCAL MACHINE
# ============================================================
# Usage:
#   chmod +x deploy-local.sh
#   ./deploy-local.sh <VPS_IP> <SSH_USER>
#
# Example:
#   ./deploy-local.sh 203.0.113.42 root
# ============================================================
set -euo pipefail

VPS_IP="${1:?First argument must be the VPS IP address}"
SSH_USER="${2:-root}"
REMOTE="${SSH_USER}@${VPS_IP}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)/rw-trader"

echo "=== RW-Trader deploy: local → ${REMOTE} ==="

# ── 1. Package the project ──────────────────────────────────
echo "[1/4] Packaging source..."
tar -czf /tmp/rw-trader-src.tar.gz -C "$(dirname "$PROJECT_DIR")" rw-trader/

# ── 2. Upload source + remote bootstrap ────────────────────
echo "[2/4] Uploading to ${REMOTE}..."
scp /tmp/rw-trader-src.tar.gz "${REMOTE}:/tmp/rw-trader-src.tar.gz"
scp "$(dirname "$0")/deploy-remote.sh" "${REMOTE}:/tmp/deploy-remote.sh"

# ── 3. Run bootstrap on VPS ────────────────────────────────
echo "[3/4] Running remote setup (this takes ~5 min for first Rust build)..."
ssh "${REMOTE}" "bash /tmp/deploy-remote.sh"

# ── 4. Done ────────────────────────────────────────────────
echo ""
echo "=== Deployment complete ==="
echo ""
echo "  Web UI:   http://${VPS_IP}:8080"
echo "  Events:   http://${VPS_IP}:8080/events"
echo "  Status:   http://${VPS_IP}:8080/status"
echo "  Authority:http://${VPS_IP}:8080/authority"
echo "  Suggest:  http://${VPS_IP}:8080/suggestions"
echo ""
echo "  Mode: LIVE_TRADE=false (monitor only, no orders)"
echo "  Authority: OFF (no execution from UI)"
echo ""
echo "  To check service: ssh ${REMOTE} systemctl status rw-trader"
echo "  To view logs:     ssh ${REMOTE} journalctl -u rw-trader -f"
echo "  To enable trading:  edit /opt/rw-trader/.env  →  LIVE_TRADE=true"
echo "                      then: ssh ${REMOTE} systemctl restart rw-trader"
