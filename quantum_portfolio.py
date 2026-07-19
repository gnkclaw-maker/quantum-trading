#!/usr/bin/env python3
"""
Quantum-inspired discrete portfolio optimizer.

Formulates mean-variance portfolio selection as a QUBO problem (the same math
fed to D-Wave annealers / QAOA circuits), with realistic constraints:

  - budget (total capital deployed)
  - cardinality (max number of positions)
  - sector caps (max capital per sector)
  - transaction-cost-aware rebalancing vs current holdings

Solvers:
  - exact brute force (n <= 24 binary vars)
  - SQA: path-integral simulated quantum annealing (built-in, no deps)
  - neal: D-Wave's classical simulated annealer   (pip install dwave-neal)
  - dwave: actual quantum annealing hardware via D-Wave Leap free tier
           (pip install dwave-system + free API token from cloud.dwavesys.com)

Examples:
  python3 quantum_portfolio.py --universe stocks
  python3 quantum_portfolio.py --universe crypto --budget 8000 --lambda 1.5
  python3 quantum_portfolio.py --universe sp500 --screen 60 --budget 50000 \
      --max-positions 10 --sector-cap 0.35
  python3 quantum_portfolio.py --tickers AAPL MSFT NVDA --solver neal
  python3 quantum_portfolio.py --universe stocks --solver dwave   # real QPU
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd

# ---------------- preset universes ----------------
UNIVERSES = {
    "stocks": {
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
                    "JPM", "XOM", "JNJ", "PG"],
        "sectors": {"AAPL": "tech", "MSFT": "tech", "NVDA": "tech",
                    "GOOGL": "tech", "AMZN": "consumer", "META": "tech",
                    "JPM": "financials", "XOM": "energy", "JNJ": "health",
                    "PG": "consumer"},
    },
    "crypto": {
        "tickers": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
                    "ADA-USD", "AVAX-USD", "LINK-USD"],
        "sectors": {"BTC-USD": "l1", "ETH-USD": "l1", "SOL-USD": "l1",
                    "BNB-USD": "exchange", "XRP-USD": "payments",
                    "ADA-USD": "l1", "AVAX-USD": "l1", "LINK-USD": "infra"},
    },
    "commodities": {
        # liquid ETFs as tradable proxies
        "tickers": ["GLD", "SLV", "USO", "UNG", "COPX", "DBA", "PPLT", "CPER"],
        "sectors": {"GLD": "precious", "SLV": "precious", "PPLT": "precious",
                    "USO": "energy", "UNG": "energy", "COPX": "industrial",
                    "CPER": "industrial", "DBA": "agriculture"},
    },
}

SEED = 7
rng = np.random.default_rng(SEED)


# ---------------- universes ----------------
def get_sp500():
    """Fetch current S&P 500 constituents + GICS sectors from Wikipedia."""
    import io, urllib.request
    req = urllib.request.Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (quantum-trading; research bot)"})
    html = urllib.request.urlopen(req, timeout=30).read()
    df = pd.read_html(io.BytesIO(html))[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    sectors = dict(zip(tickers, df["GICS Sector"]))
    return tickers, sectors


# ---------------- data ----------------
def get_data(tickers, lookback):
    """Returns px, mu, cov, surviving_tickers (drops tickers with gaps)."""
    import yfinance as yf
    px = yf.download(tickers, period=lookback, interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    px = px.dropna(axis=1, thresh=max(30, int(0.98 * len(px)))).dropna()
    alive = [t for t in tickers if t in px.columns]
    dropped = [t for t in tickers if t not in alive]
    if dropped:
        print(f"  (dropped {len(dropped)} tickers with insufficient history)")
    px = px[alive]
    rets = px.pct_change().dropna()
    return px, rets.mean().values * 252, rets.cov().values * 252, alive


def rsi(px, period=14):
    """Wilder RSI on a price series (last value)."""
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def technical_score(tickers, px, mu, cov):
    """Composite technical score per ticker: momentum trend, RSI-50 distance,
    risk-adjusted return. Returns (scores array, details dict)."""
    vol = np.sqrt(np.diag(cov))
    sharpe = np.divide(mu, vol, out=np.full_like(mu, -9e9), where=vol > 0)
    scores, details = np.zeros(len(tickers)), {}
    for j, t in enumerate(tickers):
        p = px[t]
        sma50 = p.rolling(50).mean().iloc[-1]
        sma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
        trend = (p.iloc[-1] / sma50 - 1) * 100
        golden = 1.0 if (not np.isnan(sma200) and sma50 > sma200) else (
                 -1.0 if not np.isnan(sma200) else 0.0)
        r = rsi(p)
        rsi_term = (r - 50) / 25          # >0 overbought-momentum zone, clipped
        mom = np.tanh(sharpe[j])          # bounded risk-adjusted return
        score = 0.45 * mom + 0.35 * np.tanh(trend / 10) + 0.2 * np.clip(rsi_term, -1, 1) + 0.1 * golden
        scores[j] = score
        details[t] = {"rsi": r, "trend_50d": trend,
                      "above_200dma": bool(golden > 0) if not np.isnan(sma200) else None,
                      "tech_score": score}
    return scores, details


def fundamental_score(tickers):
    """Composite fundamental score via yfinance .info: growth, profitability,
    valuation, leverage. Missing data -> neutral 0."""
    import yfinance as yf
    scores, details = {}, {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rev_g = info.get("revenueGrowth")          # e.g. 0.15
            pm = info.get("profitMargins")             # e.g. 0.25
            roe = info.get("returnOnEquity")
            pe = info.get("forwardPE")
            peg = info.get("pegRatio")
            dte = info.get("debtToEquity")             # percent
            f = 0.0; n = 0
            if rev_g is not None: f += 0.30 * np.tanh(rev_g / 0.20); n += 1
            if pm is not None:    f += 0.25 * np.tanh(pm / 0.20);    n += 1
            if roe is not None:   f += 0.20 * np.tanh(roe / 0.20);   n += 1
            if pe:                f += 0.10 * np.tanh((25 - pe) / 25); n += 1
            if peg:               f += 0.10 * np.tanh((2 - peg) / 2);  n += 1
            if dte is not None:   f += 0.05 * np.tanh((100 - dte) / 100); n += 1
            s = float(f) if n else 0.0
            scores[t] = s
            details[t] = {"rev_growth": rev_g, "profit_margin": pm,
                          "forward_pe": pe, "debt_to_eq": dte, "fund_score": s}
        except Exception:
            scores[t] = 0.0
            details[t] = {"fund_score": 0.0}
    return np.array([scores[t] for t in tickers]), details


def screen(tickers, px, mu, cov, n_keep, w_tech=0.5, w_fund=0.5):
    """Rank universe by combined technical + fundamental score, keep top n_keep.
    Shrinks the universe classically so the QUBO handles the constrained
    combinatorics on the quality names."""
    print(f"  scoring {len(tickers)} tickers (technical {w_tech:.0%} + "
          f"fundamental {w_fund:.0%})...")
    tech, tech_d = technical_score(tickers, px, mu, cov)
    fund, fund_d = fundamental_score(tickers)
    combined = w_tech * tech + w_fund * fund
    details = {}
    for j, t in enumerate(tickers):
        details[t] = {**tech_d[t], **fund_d[t], "combined": float(combined[j])}
    keep = np.sort(np.argsort(-combined)[:n_keep])
    ranked = sorted(details.items(), key=lambda kv: -kv[1]["combined"])
    return keep, ranked


# ---------------- QUBO construction ----------------
def build_qubo(mu, cov, budget_lots, lam, max_positions=None, sector_cap=None,
               sectors=None, tickers=None, tcost_bps=0.0, current_lots=None):
    """Binary vars x_{i,b}: bit b of asset i -> lots_i = sum_b 2^b x_{i,b}.
    Energy = -mu.lots + lam * lots'Cov lots
           + P_budget*(sum lots - B)^2
           + P_card*(sum pos - K)^2            (pos_i = any lots)
           + P_sector*(sector_lots - cap)^2    (violated part approximated)
           + transaction costs vs current_lots (linear approx)
    Returns Q (n x n symmetric), meta dict."""
    n_stocks = len(mu)
    bits = 2                                # 0..3 lots per asset
    n = n_stocks * bits
    w = np.tile([1.0, 2.0], n_stocks)       # lot weight per binary var
    stock_of = np.repeat(np.arange(n_stocks), bits)

    mu_var = mu[stock_of]
    cov_var = cov[np.ix_(stock_of, stock_of)]

    # --- core mean-variance objective
    Q = lam * np.outer(w, w) * cov_var
    Q[np.diag_indices(n)] -= mu_var * w

    scale_ref = max(np.abs(mu_var * w).sum(), 1e-9)

    # --- budget penalty
    P_b = 10 * scale_ref / max(budget_lots, 1)
    Q += P_b * np.outer(w, w)
    Q[np.diag_indices(n)] -= 2 * P_b * budget_lots * w

    # --- cardinality penalty: pos indicator ~ (lots>0); use OR of bits
    if max_positions is not None:
        # pos_i <=> x_i0 OR x_i1 ; penalize (sum_i pos_i - K)^2 using
        # pos_i approx = (x_i0 + x_i1) with extra penalty on both-bits
        P_k = 5 * scale_ref / max(max_positions, 1)
        pos = np.zeros(n)
        pos[0::2] = 1.0; pos[1::2] = 1.0
        Q += P_k * np.outer(pos, pos)
        Q[np.diag_indices(n)] -= 2 * P_k * max_positions * pos
        # discourage counting both bits twice (pos is binary OR)
        for i in range(n_stocks):
            Q[i * 2, i * 2 + 1] -= P_k * 0.5
            Q[i * 2 + 1, i * 2] -= P_k * 0.5

    # --- sector caps
    if sector_cap is not None and sectors and tickers:
        cap_lots = sector_cap * budget_lots
        sec_of = [sectors[t] for t in tickers]
        for sec in set(sec_of):
            idx = [i * 2 + b for i in range(n_stocks) if sec_of[i] == sec
                   for b in range(bits)]
            if not idx:
                continue
            v = np.zeros(n)
            v[idx] = w[idx]
            P_s = 5 * scale_ref / max(cap_lots, 1)
            # hinge approximated as quadratic around cap
            Q += P_s * np.outer(v, v)
            Q[np.diag_indices(n)] -= 2 * P_s * cap_lots * v

    # --- transaction costs (linear: cost per lot changed vs current)
    if current_lots is not None and tcost_bps > 0:
        cur = np.asarray(current_lots, dtype=float)
        # cost ~ tcost * |new - old|; linearized: penalize increases, reward holding
        c = tcost_bps / 1e4 * LOT_VALUE_DEFAULT * np.ones(n_stocks)
        lin = c[stock_of] * w
        Q[np.diag_indices(n)] += lin           # cost of buying a lot
        for i in range(n_stocks):
            for b in range(bits):
                if cur[i] >= (2 ** b):
                    Q[i * 2 + b, i * 2 + b] -= 2 * c[i] * (2 ** b)  # rebate for keeping

    meta = {"w": w, "bits": bits, "n_stocks": n_stocks}
    return Q, meta


LOT_VALUE_DEFAULT = 1000.0


# ---------------- solvers ----------------
def energy(Q, x):
    return x @ Q @ x


def brute_force(Q):
    n = Q.shape[0]
    N = 2 ** n
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    e = np.einsum('bi,ij,bj->b', bits, Q, bits)
    k = np.argmin(e)
    return bits[k], e[k]


def simulated_quantum_anneal(Q, n_trotter=12, sweeps=6000, T0=1.0, T1=0.01,
                             gamma0=3.0, gamma1=0.15):
    """Path-integral SQA: Trotter replicas coupled in imaginary time;
    transverse field annealed high->low mimics quantum tunneling."""
    n = Q.shape[0]
    X = rng.choice([-1, 1], size=(n_trotter, n))
    J = Q / 4.0
    h = Q.sum(axis=1) / 2.0
    J = J.copy(); np.fill_diagonal(J, 0)

    scale = np.abs(Q).sum() / n
    T0, T1, gamma0, gamma1 = (T0 * scale, T1 * scale,
                              gamma0 * scale, gamma1 * scale)
    best_e, best_x = np.inf, None
    for sweep in range(sweeps):
        t = sweep / sweeps
        T = T0 * (T1 / T0) ** t
        gamma = gamma0 * (gamma1 / gamma0) ** t
        Jp = -T * np.log(np.tanh(gamma / (n_trotter * T))) / 2.0
        for _ in range(n):
            p = rng.integers(n_trotter); i = rng.integers(n)
            dE = 2 * X[p, i] * (h[i] + J[i] @ X[p])
            dE += 2 * X[p, i] * Jp * (X[(p - 1) % n_trotter, i] +
                                      X[(p + 1) % n_trotter, i])
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                X[p, i] *= -1
        Xb = ((X + 1) // 2).astype(float)
        for p in range(n_trotter):
            e = energy(Q, Xb[p])
            if e < best_e:
                best_e, best_x = e, Xb[p].copy()
    return best_x, best_e


def solve_neal(Q, num_reads=200):
    """D-Wave's classical simulated annealer (pip install dwave-neal)."""
    import neal
    n = Q.shape[0]
    qubo = {(i, j): Q[i, j] for i in range(n) for j in range(n)
            if Q[i, j] != 0}
    ss = neal.SimulatedAnnealingSampler().sample_qubo(
        qubo, num_reads=num_reads, seed=SEED)
    best = ss.first
    x = np.array([best.sample[i] for i in range(n)], dtype=float)
    return x, energy(Q, x)


def solve_dwave(Q, num_reads=100):
    """Actual D-Wave quantum annealer via Leap (needs free API token)."""
    from dwave.system import EmbeddingComposite, DWaveSampler
    n = Q.shape[0]
    qubo = {(i, j): Q[i, j] for i in range(n) for j in range(n)
            if Q[i, j] != 0}
    sampler = EmbeddingComposite(DWaveSampler())
    ss = sampler.sample_qubo(qubo, num_reads=num_reads)
    best = ss.first
    x = np.array([best.sample[i] for i in range(n)], dtype=float)
    return x, energy(Q, x)


# ---------------- reporting ----------------
def describe(x, meta, tickers, mu, cov, lot_value):
    bits, w = meta["bits"], meta["w"]
    lots = x.reshape(-1, bits) @ (2.0 ** np.arange(bits))
    port = pd.DataFrame({"ticker": tickers, "lots": lots.astype(int)})
    port["$"] = port.lots * lot_value
    port = port[port.lots > 0]
    tot = lots.sum()
    if tot > 0:
        weights = lots / tot
        ann_ret = float(weights @ mu)
        ann_vol = float(np.sqrt(weights @ cov @ weights))
    else:
        ann_ret = ann_vol = 0.0
    return port, int(tot), ann_ret, ann_vol


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", choices=list(UNIVERSES) + ["sp500"], default="stocks")
    ap.add_argument("--tickers", nargs="+", help="custom tickers (overrides universe)")
    ap.add_argument("--screen", type=int, default=None,
                    help="pre-screen universe to top N by tech+fundamental score")
    ap.add_argument("--tech-weight", type=float, default=0.5,
                    help="weight of technical score in screening (rest = fundamentals)")
    ap.add_argument("--fractional", action="store_true",
                    help="fractional shares: allocate exact $ instead of whole lots")
    ap.add_argument("--min-position", type=float, default=100,
                    help="min $ per position (fractional mode)")
    ap.add_argument("--lookback", default="1y")
    ap.add_argument("--budget", type=float, default=12000, help="capital in $")
    ap.add_argument("--lot-value", type=float, default=1000)
    ap.add_argument("--lots-per-stock", type=int, default=3, choices=[1, 3, 7])
    ap.add_argument("--lambda", dest="lam", type=float, default=2.5,
                    help="risk aversion")
    ap.add_argument("--save", default=None, help="save chosen portfolio to JSON")
    ap.add_argument("--no-brute-force", action="store_true")
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--sector-cap", type=float, default=None,
                    help="max fraction of budget per sector, e.g. 0.5")
    ap.add_argument("--tcost-bps", type=float, default=0.0,
                    help="transaction cost per side, basis points")
    ap.add_argument("--current-holdings", default=None,
                    help='JSON like {"AAPL":2,"XOM":1} (lots) for rebalancing')
    ap.add_argument("--solver", choices=["sqa", "neal", "dwave"], default="sqa")
    ap.add_argument("--restarts", type=int, default=12)
    args = ap.parse_args()

    global LOT_VALUE_DEFAULT
    LOT_VALUE_DEFAULT = args.lot_value

    uni = UNIVERSES.get(args.universe)
    if args.universe == "sp500":
        print("Fetching S&P 500 constituents...")
        tickers, sectors = get_sp500()
    else:
        tickers = args.tickers or uni["tickers"]
        sectors = uni["sectors"] if not args.tickers else None

    # auto lot size: make lots granular enough relative to capital
    if args.fractional:
        target_lots = max(10 * args.max_positions if args.max_positions else 30, 30)
        args.lot_value = max(args.min_position,
                             round(args.budget / target_lots, 2))
    budget_lots = int(args.budget // args.lot_value)
    print(f"Universe: {args.universe} ({len(tickers)} assets) | "
          f"budget ${args.budget:,.0f} = {budget_lots} lots x ${args.lot_value:,.0f} | "
          f"lambda={args.lam}"
          + (f" | max {args.max_positions} positions" if args.max_positions else "")
          + (f" | sector cap {args.sector_cap:.0%}" if args.sector_cap else ""))

    print(f"Downloading {args.lookback} of daily data...")
    px, mu, cov, tickers = get_data(tickers, args.lookback)
    if sectors:
        sectors = {t: sectors[t] for t in tickers}
    ranked = None
    if args.screen and len(tickers) > args.screen:
        keep, ranked = screen(tickers, px, mu, cov, args.screen,
                              w_tech=args.tech_weight, w_fund=1 - args.tech_weight)
        tickers = [tickers[i] for i in keep]
        mu, cov = mu[keep], cov[np.ix_(keep, keep)]
        px = px[tickers]
        if sectors:
            sectors = {t: sectors[t] for t in tickers}
        ranked = ranked[:args.screen]
        print(f"Pre-screened to top {len(tickers)}:")
        for t, d in ranked[:10]:
            print(f"   {t:6s} score {d['combined']:+.2f} "
                  f"(tech {d['tech_score']:+.2f}, fund {d['fund_score']:+.2f}, "
                  f"RSI {d['rsi']:.0f}, trend {d['trend_50d']:+.1f}%)")
        if len(ranked) > 10:
            print(f"   ... and {len(ranked)-10} more")

    current = None
    if args.current_holdings:
        ch = json.loads(args.current_holdings)
        current = [ch.get(t, 0) for t in tickers]

    Q, meta = build_qubo(mu, cov, budget_lots, args.lam,
                         max_positions=args.max_positions,
                         sector_cap=args.sector_cap,
                         sectors=sectors, tickers=tickers,
                         tcost_bps=args.tcost_bps, current_lots=current)
    n = Q.shape[0]
    print(f"QUBO: {n} binary variables\n")

    ref = None
    if n <= 24 and not args.no_brute_force:
        x_bf, e_bf = brute_force(Q)
        ref = (x_bf, e_bf)
        print(f"[exact brute force: {2**n:,} states]")
        port, nl, r, v = describe(x_bf, meta, tickers, mu, cov, args.lot_value)
        print(port.to_string(index=False))
        print(f"  lots {nl}/{budget_lots} | E[ret] {r:+.1%} | vol {v:.1%} | "
              f"Sharpe~{r/v:.2f}\n")

    print(f"[solver: {args.solver}]")
    if args.solver == "sqa":
        runs = [simulated_quantum_anneal(Q) for _ in range(args.restarts)]
        x, e = min(runs, key=lambda r: r[1])
        if ref:
            hits = sum(1 for _, ee in runs if np.isclose(ee, ref[1]))
            print(f"  global optimum hit in {hits}/{args.restarts} restarts")
    elif args.solver == "neal":
        x, e = solve_neal(Q)
    else:
        x, e = solve_dwave(Q)

    port, nl, r, v = describe(x, meta, tickers, mu, cov, args.lot_value)
    print(port.to_string(index=False))
    print(f"  lots {nl}/{budget_lots} | E[ret] {r:+.1%} | vol {v:.1%} | "
          f"Sharpe~{r/v:.2f}")
    if ref:
        gap = (e - ref[1]) / abs(ref[1]) * 100
        tag = "found global optimum" if np.isclose(e, ref[1]) else "near-optimal"
        print(f"\nEnergy gap vs exact optimum: {gap:+.3f}%  ({tag})")

    if args.save:
        out = {t: int(l) for t, l in zip(tickers, x.reshape(-1, meta['bits']) @ (2.0 ** np.arange(meta['bits'])))}
        out = {t: l for t, l in out.items() if l > 0}
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved portfolio -> {args.save}")


if __name__ == "__main__":
    main()
