#!/usr/bin/env bash
# ============================================================
# deploy-remote.sh — runs ON THE VPS via SSH
# Idempotent: safe to re-run for updates.
# ============================================================
set -euo pipefail

APP_DIR="/opt/rw-trader"
SERVICE="rw-trader"
BINARY_NAME="rw-trader"
DATA_DIR="${APP_DIR}/data"
LOG_DIR="/var/log/rw-trader"

echo ""
echo "=== [1/8] System packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    pkg-config \
    libssl-dev \
    curl \
    ufw \
    > /dev/null

echo "=== [2/8] Rust toolchain ==="
if ! command -v cargo &>/dev/null; then
    echo "  Installing Rust..."
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    # Make cargo available for the rest of this script
    source "$HOME/.cargo/env"
else
    echo "  Rust already installed: $(rustc --version)"
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

echo "=== [3/8] Unpack project ==="
mkdir -p "${APP_DIR}" "${DATA_DIR}" "${LOG_DIR}"
tar -xzf /tmp/rw-trader-src.tar.gz -C /tmp/
rsync -a --delete /tmp/rw-trader/ "${APP_DIR}/src-build/"

echo "=== [4/8] Build release binary ==="
cd "${APP_DIR}/src-build"
cargo build --release 2>&1 | tail -5
cp target/release/${BINARY_NAME} "${APP_DIR}/${BINARY_NAME}"
echo "  Binary: $(ls -lh ${APP_DIR}/${BINARY_NAME} | awk '{print $5, $9}')"

echo "=== [5/8] Write .env (safe: only if not already present) ==="
ENV_FILE="${APP_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    cat > "${ENV_FILE}" << 'ENVEOF'
# ── Exchange credentials ───────────────────────────────────
# Replace with your actual Binance API key and secret.
# Use Binance Testnet keys for safe testing:
#   https://testnet.binance.vision
BINANCE_API_KEY=your_binance_testnet_api_key_here
BINANCE_API_SECRET=your_binance_testnet_api_secret_here
BINANCE_REST_URL=https://testnet.binance.vision
BINANCE_WS_URL=wss://testnet.binance.vision/ws

# ── Symbol ─────────────────────────────────────────────────
SYMBOL=BTCUSDT
BNB_PRICE_USD=0

# ── Trading mode ───────────────────────────────────────────
# SAFE DEFAULT: false = monitor-only, no orders placed.
# Set to true only after verifying your config on testnet.
LIVE_TRADE=false

# ── Data storage ───────────────────────────────────────────
EVENT_DB_PATH=/opt/rw-trader/data/rw-trader-events.db

# ── Web UI ─────────────────────────────────────────────────
# 0.0.0.0 makes UI accessible from the internet on port 8080.
# Access at: http://<VPS_IP>:8080
WEB_UI_ADDR=0.0.0.0:8080

# ── Reconciliation ──────────────────────────────────────────
RECONCILE_INTERVAL_SECS=2

# ── Circuit breaker ────────────────────────────────────────
CB_MAX_ATTEMPTS=10
CB_MAX_REJECTS=3
CB_MAX_ERRORS=3
CB_MAX_SLIPPAGE=2

# ── Watchdog ───────────────────────────────────────────────
WD_ACK_TIMEOUT_SECS=10
WD_CANCEL_TIMEOUT_SECS=5
WD_REPLACE_TIMEOUT_SECS=15

# ── Risk limits ────────────────────────────────────────────
RISK_MAX_QTY=0.001
RISK_MAX_DAILY_LOSS=10.0
RISK_MAX_DRAWDOWN=20.0
RISK_MAX_SPREAD_BPS=10.0
RISK_COOLDOWN_SECS=300
RISK_FEED_STALE_SECS=5

# ── Signal thresholds ──────────────────────────────────────
SIGNAL_QTY=0.001
SIGNAL_MOMENTUM_THRESH=0.00005
SIGNAL_IMBALANCE_THRESH=0.10
SIGNAL_MAX_SPREAD_BPS=5.0
SIGNAL_STOP_LOSS_PCT=0.0020
SIGNAL_TAKE_PROFIT_PCT=0.0040
SIGNAL_MAX_HOLD_SECS=120

# ── Logging ────────────────────────────────────────────────
RUST_LOG=rw_trader=info,warn
ENVEOF
    echo "  .env written (EDIT BEFORE ENABLING LIVE TRADING)"
else
    echo "  .env already exists — skipping (preserving credentials)"
    # Always ensure WEB_UI_ADDR is set for external access
    if ! grep -q "^WEB_UI_ADDR" "${ENV_FILE}"; then
        echo "WEB_UI_ADDR=0.0.0.0:8080" >> "${ENV_FILE}"
        echo "  Added WEB_UI_ADDR to existing .env"
    fi
fi
chmod 600 "${ENV_FILE}"

echo "=== [6/8] systemd service ==="
cat > "/etc/systemd/system/${SERVICE}.service" << SVCEOF
[Unit]
Description=RW-Trader — Binance algorithmic trading bot
Documentation=https://github.com/rw-trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/${BINARY_NAME}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE}

# Safety: authority defaults to OFF in the binary.
# No orders possible without explicit operator action in /authority.

# Resource limits — prevent runaway memory
MemoryMax=512M
TasksMax=64

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable "${SERVICE}"

echo "=== [7/8] Firewall ==="
# Allow SSH (critical — must come before enable)
ufw allow 22/tcp comment 'SSH' > /dev/null
# Allow web UI
ufw allow 8080/tcp comment 'rw-trader web UI' > /dev/null
# Block everything else inbound
ufw --force enable > /dev/null
echo "  ufw status:"
ufw status numbered | grep -E "22|8080" || true

echo "=== [8/8] Start / restart service ==="
if systemctl is-active --quiet "${SERVICE}"; then
    echo "  Restarting existing service..."
    systemctl restart "${SERVICE}"
else
    echo "  Starting service for first time..."
    systemctl start "${SERVICE}"
fi

sleep 2
echo ""
echo "  Service status:"
systemctl status "${SERVICE}" --no-pager -l | head -15

echo ""
echo "==================================================================="
echo " RW-Trader is running."
echo ""
echo " NEXT STEPS:"
echo ""
echo " 1. Edit credentials:"
echo "    nano ${ENV_FILE}"
echo "    → Replace BINANCE_API_KEY and BINANCE_API_SECRET"
echo "    → Use Binance Testnet keys first"
echo ""
echo " 2. Restart after editing .env:"
echo "    systemctl restart ${SERVICE}"
echo ""
echo " 3. View logs:"
echo "    journalctl -u ${SERVICE} -f"
echo ""
echo " 4. The web UI is available on port 8080."
echo "    AuthorityMode starts as OFF — no execution possible"
echo "    until you visit /authority and switch to ASSIST or AUTO."
echo ""
echo "==================================================================="
