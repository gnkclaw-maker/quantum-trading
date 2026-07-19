# quantum-trading

**Quantum-inspired portfolio optimization on classical hardware** — the one
piece of "quantum trading" that actually works today.

Formulates mean-variance portfolio selection as a **QUBO** problem — the exact
same math fed to D-Wave quantum annealers and QAOA circuits — and solves it
with simulated quantum annealing. Optionally runs on **real D-Wave quantum
hardware** via the free Leap tier.

> **Context:** as of 2026, no quantum computer beats classical hardware for
> trading. What works now is quantum-*inspired* math on classical machines.
> Every "Quantum AI" trading app promising returns is a scam — this repo is
> the honest version of the idea.

## Features

- 📈 **Three preset universes**: stocks, crypto, commodities (ETF proxies) — or bring your own tickers
- 🧮 **QUBO formulation** with realistic constraints:
  - capital budget (discrete lots)
  - max number of positions (cardinality)
  - sector caps
  - transaction-cost-aware rebalancing vs current holdings
- ⚛️ **Solvers**:
  - `sqa` — path-integral simulated quantum annealing (built-in, zero deps)
  - `neal` — D-Wave's classical simulated annealer
  - `dwave` — actual quantum annealing hardware (D-Wave Leap free tier)
- ✅ **Verified**: brute-force exact optimum computed for small problems so
  you can see the annealer's optimality gap (typically <1%)

## Install

```bash
pip install numpy pandas yfinance
# optional solvers:
pip install dwave-neal        # for --solver neal
pip install dwave-system      # for --solver dwave (real QPU, needs free Leap token)
```

## Quick start

```bash
# stocks, $12k budget
python3 quantum_portfolio.py --universe stocks

# crypto, $8k, max 3 positions
python3 quantum_portfolio.py --universe crypto --budget 8000 --max-positions 3

# commodities with sector caps + rebalancing from current holdings
python3 quantum_portfolio.py --universe commodities --sector-cap 0.5 \
    --tcost-bps 10 --current-holdings '{"GLD":2,"USO":1}'

# your own tickers
python3 quantum_portfolio.py --tickers AAPL MSFT NVDA AMD --budget 20000

# run on a real quantum annealer (free account at cloud.dwavesys.com)
python3 quantum_portfolio.py --universe stocks --solver dwave
```

## Example output (stocks)

```
[exact brute force: 1,048,576 states]
  AAPL 2, NVDA 2, GOOGL 3, AMZN 1, XOM 1, PG 3
  lots 12/12 | E[ret] +44.7% | vol 10.8% | Sharpe~4.12

[solver: sqa]
  global optimum hit in 3/12 restarts
Energy gap vs exact optimum: +0.074%  (near-optimal)
```

## How it works

Each asset gets 2 binary variables (0–3 lots). The energy function:

```
E(x) = -μ·lots + λ·lots'Σlots + P_budget·(Σlots - B)²
       + P_card·(Σpositions - K)² + P_sector·(sector_lots - cap)²
       + transaction_costs
```

- **Brute force** evaluates all 2ⁿ portfolios (feasible to ~24 vars) → ground truth
- **SQA** runs Trotter replicas coupled in imaginary time with an annealed
  transverse field, mimicking quantum tunneling through energy barriers —
  no enumeration, scales to hundreds of variables
- The same QUBO matrix drops straight into D-Wave's sampler → real QPU run

At 20 variables brute force is trivial. At 200+ assets it is impossible —
that's where annealing (and eventually real quantum advantage) matters.

## Honest limitations

- Optimizes **allocation, not signal** — it sizes positions given expected
  returns; it does not predict prices
- Reported returns/Sharpe are **in-sample**; not investment advice
- Constraints are soft (penalty method) — near-boundary solutions can
  slightly exceed a constraint; tune penalties or filter results
- Real QPU runs are educational, not faster — current hardware doesn't beat
  classical annealers at this scale

## Roadmap

- [ ] CVaR / expected-shortfall objective
- [ ] Walk-forward backtest harness
- [ ] QAOA circuit export (Qiskit) for gate-based hardware
- [ ] Benchmark suite: SQA vs neal vs D-Wave QPU vs Gurobi

## License

MIT
