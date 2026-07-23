#!/usr/bin/env python3
"""
Daily pre-market pipeline:
  1. Screen S&P 500 (technical + fundamental)
  2. QUBO/anneal allocation sized to Alpaca paper account equity
  3. Send brief to Telegram (topic or DM)
  4. If --trade: place market orders at next open (paper account)

Designed to be run by cron at 13:20 UTC Mon-Fri (before 9:30 ET open).
Orders submitted while market is closed become market-on-open orders.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
import quantum_portfolio as qp
import capitol_trades as ct

ENV = Path.home() / ".openclaw/credentials/alpaca.env"

def load_env():
    for line in ENV.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def headers():
    return {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET"]}

BASE = "https://paper-api.alpaca.markets"

def account():
    return requests.get(f"{BASE}/v2/account", headers=headers(), timeout=30).json()

def clock():
    return requests.get(f"{BASE}/v2/clock", headers=headers(), timeout=30).json()

def positions():
    return requests.get(f"{BASE}/v2/positions", headers=headers(), timeout=30).json()

def order(symbol, qty, side="buy"):
    payload = {"symbol": symbol, "qty": round(qty, 4), "side": side,
               "type": "market", "time_in_force": "day"}
    r = requests.post(f"{BASE}/v2/orders", headers=headers(), json=payload, timeout=30)
    return r.status_code, r.json()


def run_pipeline(budget, max_positions, screen_n, sector_cap, lam, tech_weight,
                 restarts, congress_weight=0.0):
    tickers, sectors = qp.get_sp500()
    px, mu, cov, tickers = qp.get_data(tickers, "1y")
    sectors = {t: sectors[t] for t in tickers}

    congress_trades, extra = [], None
    if congress_weight:
        try:
            congress_trades = ct.fetch_recent_trades()
            signals = ct.congress_buy_signals(congress_trades)
            extra = {t: congress_weight * s for t, s in signals.items()}
            print(f"  congress signal: {len(congress_trades)} trades, "
                  f"{len(signals)} tickers with buys (weight {congress_weight})",
                  file=sys.stderr)
        except Exception as e:
            print(f"  congress signal unavailable: {e}", file=sys.stderr)

    keep, ranked = qp.screen(tickers, px, mu, cov, min(screen_n, len(tickers)),
                             w_tech=tech_weight, w_fund=1 - tech_weight,
                             extra=extra)
    tickers = [tickers[i] for i in keep]
    mu, cov = mu[keep], cov[np.ix_(keep, keep)]
    px = px[tickers]
    sectors = {t: sectors[t] for t in tickers}

    target_lots = max(10 * max_positions, 30)
    lot_value = max(100, round(budget / target_lots, 2))
    budget_lots = int(budget // lot_value)

    Q, meta = qp.build_qubo(mu, cov, budget_lots, lam,
                            max_positions=max_positions, sector_cap=sector_cap,
                            sectors=sectors, tickers=tickers)
    runs = [qp.simulated_quantum_anneal(Q) for _ in range(restarts)]
    x, e = min(runs, key=lambda r: r[1])
    port, nl, r, v = qp.describe(x, meta, tickers, mu, cov, lot_value)
    return port, ranked, px, (r, v), (budget, budget_lots, lot_value), congress_trades


def build_brief(port, ranked, px, stats, sizing, today, congress_trades=None):
    r, v = stats
    budget, budget_lots, lot_value = sizing
    det = dict(ranked)
    lines = [f"**📊 Pre-market brief — {today}**\n",
             f"Universe: S&P 500 → top {len(ranked)} | capital ${budget:,.0f} | "
             f"max positions, sector-capped\n",
             "**Proposed allocation** (QUBO/SQA)"]
    trades = []
    for _, row in port.iterrows():
        t = row.ticker
        d = det.get(t, {})
        last = float(px[t].iloc[-1])
        shares = row["$"] / last
        trades.append({"symbol": t, "qty": round(shares, 4),
                       "notional": row["$"], "last": last})
        cg = f" | 🏛️ {d['congress']:+.2f}" if d.get("congress") else ""
        lines.append(
            f"• **{t}** — ${row['$']:,.0f} (~{shares:.2f} sh @ {last:,.2f})"
            f" | score {d.get('combined', 0):+.2f} | RSI {d.get('rsi', 0):.0f}"
            f" | trend {d.get('trend_50d', 0):+.1f}%{cg}")
    deployed = port["$"].sum()
    lines.append(f"\nDeployed ${deployed:,.0f} of ${budget:,.0f} "
                 f"({deployed / budget:.0%}) | model E[ret] {r:+.1%}, "
                 f"vol {v:.1%}, Sharpe ~{r / v:.2f} (in-sample)")
    if congress_trades:
        notable = ct.summarize_buys(congress_trades, limit=5)
        if notable:
            lines.append("\n**🏛️ Recent Congress purchases** (STOCK Act disclosures)")
            for t in notable:
                lines.append(
                    f"• **{t['ticker']}** — {t['politician']} ({t.get('chamber', '?')}) "
                    f"bought {t['amount_range']} on {t['trade_date']}")
    lines.append("\n**Screen leaderboard**")
    for t, d in ranked[:8]:
        cg = f" · 🏛️ {d['congress']:+.2f}" if d.get("congress") else ""
        lines.append(f"• {t}: {d['combined']:+.2f} (tech {d['tech_score']:+.2f}"
                     f" · fund {d['fund_score']:+.2f} · RSI {d['rsi']:.0f}{cg})")
    lines.append("\n_Paper trading via Alpaca. Not investment advice._")
    return "\n".join(lines), trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None,
                    help="override; default = Alpaca account cash")
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--screen", type=int, default=30)
    ap.add_argument("--sector-cap", type=float, default=0.4)
    ap.add_argument("--lam", type=float, default=2.5)
    ap.add_argument("--tech-weight", type=float, default=0.5)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--congress-weight", type=float, default=0.1,
                    help="additive weight of congress buy signal in screening "
                         "(0 = disabled)")
    ap.add_argument("--trade", action="store_true",
                    help="place market orders on paper account")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_env()
    acct = account()
    budget = args.budget or float(acct["cash"])
    clk = clock()
    today = dt.date.today().strftime("%a %Y-%m-%d")

    # deploy at most 95% of cash, keep a buffer
    budget = round(budget * 0.95, 2)

    print(f"[{dt.datetime.now():%H:%M:%S}] account cash=${float(acct['cash']):,.2f}, "
          f"market_open={clk['is_open']}", file=sys.stderr)

    port, ranked, px, stats, sizing, congress_trades = run_pipeline(
        budget, args.max_positions, args.screen, args.sector_cap, args.lam,
        args.tech_weight, args.restarts, congress_weight=args.congress_weight)
    brief, trades = build_brief(port, ranked, px, stats, sizing, today,
                                congress_trades=congress_trades)

    print(brief)
    if args.out:
        Path(args.out).write_text(brief)

    if args.trade:
        print(f"\n[{dt.datetime.now():%H:%M:%S}] placing {len(trades)} paper orders...",
              file=sys.stderr)
        results = []
        for tr in trades:
            code, resp = order(tr["symbol"], tr["qty"])
            results.append({**tr, "status": code, "order_id": resp.get("id"),
                            "error": resp.get("message")})
            print(f"  {tr['symbol']}: {code} {resp.get('id') or resp.get('message')}",
                  file=sys.stderr)
        log = Path.home() / ".openclaw/logs"
        log.mkdir(exist_ok=True)
        (log / f"trades-{dt.date.today()}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
