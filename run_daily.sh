#!/usr/bin/env bash
# Daily pre-market run: screen S&P 500 -> QUBO allocate -> brief -> paper trade.
# Invoked by cron at 13:20 UTC Mon-Fri (before 9:30 ET open).
set -euo pipefail
cd "$(dirname "$0")"
source /home/ubuntu/.openclaw/credentials/alpaca.env
export ALPACA_API_KEY ALPACA_API_SECRET

STAMP=$(date +%F)
LOG=/home/ubuntu/.openclaw/logs/daily-trade-$STAMP.log
mkdir -p /home/ubuntu/.openclaw/logs

python3 daily_trade.py \
  --trade \
  --max-positions 8 \
  --screen 30 \
  --sector-cap 0.4 \
  --out /home/ubuntu/.openclaw/logs/brief-$STAMP.md \
  >> "$LOG" 2>&1

echo "done $(date -Is)" >> "$LOG"
