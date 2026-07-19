#!/usr/bin/env python3
"""
Daily pre-market brief: technical + fundamental screening, quantum-inspired
allocation, delivered as Markdown.

Run from the quantum-trading directory:
  python3 daily_brief.py --universe sp500 --budget 25000 --max-positions 8

Cron-friendly. Output goes to stdout; the caller decides delivery
(Telegram, email, file).
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import quantum_portfolio as qp
import numpy as np


def run_brief(args):
    tickers, sectors = qp.get_sp500() if args.universe == "sp500" else (
        qp.UNIVERSES[args.universe]["tickers"], qp.UNIVERSES[args.universe]["sectors"])
    print(f"  [{dt.datetime.now():%H:%M}] downloading data for {len(tickers)} tickers...", file=sys.stderr)
    px, mu, cov, tickers = qp.get_data(tickers, args.lookback)
    if sectors:
        sectors = {t: sectors[t] for t in tickers}

    # screen on generous candidate pool, then QUBO on survivors
    keep, ranked = qp.screen(tickers, px, mu, cov,
                             min(args.screen, len(tickers)),
                             w_tech=args.tech_weight, w_fund=1 - args.tech_weight)
    tickers = [tickers[i] for i in keep]
    mu, cov = mu[keep], cov[np.ix_(keep, keep)]
    px = px[tickers]
    if sectors:
        sectors = {t: sectors[t] for t in tickers}

    # size lots to capital
    target_lots = max(10 * args.max_positions, 30)
    lot_value = max(100, round(args.budget / target_lots, 2))
    budget_lots = int(args.budget // lot_value)

    Q, meta = qp.build_qubo(mu, cov, budget_lots, args.lam,
                            max_positions=args.max_positions,
                            sector_cap=args.sector_cap,
                            sectors=sectors, tickers=tickers)
    print(f"  [{dt.datetime.now():%H:%M}] annealing ({Q.shape[0]} vars)...", file=sys.stderr)
    runs = [qp.simulated_quantum_anneal(Q) for _ in range(args.restarts)]
    x, e = min(runs, key=lambda r: r[1])

    port, nl, r, v = qp.describe(x, meta, tickers, mu, cov, lot_value)
    det = dict(ranked)
    today = dt.date.today().strftime("%a %Y-%m-%d")

    out = [f"**📊 Pre-market brief — {today}**\n"]
    out.append(f"Universe: {args.universe} → screened top {len(tickers)} | "
               f"capital ${args.budget:,.0f} | max {args.max_positions} positions\n")

    out.append("**Proposed allocation** (QUBO/SQA, risk λ=%.1f)" % args.lam)
    for _, row in port.iterrows():
        t = row.ticker
        d = det.get(t, {})
        px_last = px[t].iloc[-1]
        shares = row["$"] / px_last
        out.append(
            f"• **{t}** — ${row['$']:,.0f} (~{shares:.1f} sh @ {px_last:,.2f})"
            f" | score {d.get('combined', 0):+.2f}"
            f" | RSI {d.get('rsi', 0):.0f}"
            f" | trend {d.get('trend_50d', 0):+.1f}%")
    deployed = port["$"].sum()
    out.append(f"\nDeployed ${deployed:,.0f} of ${args.budget:,.0f} "
               f"({deployed / args.budget:.0%}) | model E[ret] {r:+.1%}, "
               f"vol {v:.1%}, Sharpe ~{r / v:.2f} (in-sample)\n")

    out.append("**Screen leaderboard** (technical + fundamental)")
    for t, d in ranked[:8]:
        out.append(f"• {t}: {d['combined']:+.2f} "
                   f"(tech {d['tech_score']:+.2f} · fund {d['fund_score']:+.2f} · "
                   f"RSI {d['rsi']:.0f} · 50d {d['trend_50d']:+.1f}%)")

    out.append("\n_Caveats: in-sample stats, not advice. Optimizes allocation, "
               "not signal. Scores from trailing data._")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500",
                    choices=["sp500"] + list(qp.UNIVERSES))
    ap.add_argument("--budget", type=float, default=25000)
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--screen", type=int, default=40)
    ap.add_argument("--sector-cap", type=float, default=0.4)
    ap.add_argument("--lam", type=float, default=2.5)
    ap.add_argument("--tech-weight", type=float, default=0.5)
    ap.add_argument("--lookback", default="1y")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--out", default=None, help="also save Markdown to file")
    args = ap.parse_args()

    brief = run_brief(args)
    print(brief)
    if args.out:
        Path(args.out).write_text(brief)
        print(f"saved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
