"""
Congress trading signals — fetches recent US politician stock trades
(STOCK Act disclosures) and turns them into a per-ticker conviction score.

Primary source: quiverquant.com/congresstrading
  (same disclosure feed as capitoltrades.com, but capitoltrades sits behind
  a Vercel bot-checkpoint that blocks datacenter IPs with HTTP 429 — verified
  2026-07-23. Quiver embeds the 300 most recent trades directly in the page
  HTML as a JS array, so no API key or JS rendering is needed.)

The capitoltrades.com HTML-scraping code path is kept as a fallback.
"""
import ast
import hashlib
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

CAPITOL_BASE = "https://www.capitoltrades.com"
QUIVER_URL = "https://www.quiverquant.com/congresstrading/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _get(url, **kw):
    """GET with curl_cffi (browser TLS fingerprint) if available, else requests."""
    try:
        from curl_cffi import requests as cr
        return cr.get(url, impersonate="chrome", timeout=kw.get("timeout", 30))
    except ImportError:
        import requests
        return requests.get(url, headers=HEADERS, **kw)


# ---------------- helpers ----------------

def _make_trade_id(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _parse_amount_midpoint(amount_range: str) -> Optional[float]:
    """Dollar midpoint from range strings like '$15,001 - $50,000' or '15K–50K'."""
    s = amount_range.strip().upper().replace(",", "").replace("$", "")
    m = re.match(r"(\d+\.?\d*)([KMB]?)\s*[–\-]\s*(\d+\.?\d*)([KMB]?)", s)
    if not m:
        return None
    mult = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    low = float(m.group(1)) * mult.get(m.group(2), 1)
    high = float(m.group(3)) * mult.get(m.group(4), 1)
    return (low + high) / 2


def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ---------------- Quiver (primary) ----------------

def _extract_js_array(html: str, varname: str) -> list:
    """Pull `let <varname> = [ ... ];` out of inline page scripts."""
    i = html.find(f"let {varname} = ")
    if i < 0:
        return []
    start = html.find("[", i)
    depth, in_str, esc, quote, end = 0, False, False, "", start
    for k, ch in enumerate(html[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch in "'\"":
                in_str, quote = True, ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
    return ast.literal_eval(html[start:end])


def fetch_quiver_trades() -> list[dict]:
    """Parse the recentTradesData array embedded in Quiver's congress page."""
    resp = _get(QUIVER_URL)
    resp.raise_for_status()
    rows = _extract_js_array(resp.text, "recentTradesData")
    trades = []
    for r in rows:
        try:
            # columns: 0 ticker, 1 asset name, 2 asset type, 3 txn type,
            # 4 amount range, 5 politician, 6 chamber, 7 party,
            # 8 filed date, 9 trade date, 10 description, 11 trade slug,
            # 12 excess return %, 13 politician sort name, 14 img, 15 member id
            ticker, asset_type, txn = r[0], r[2], r[3]
            if not ticker or ticker == "-":
                continue
            if txn == "Purchase":
                trade_type = "BUY"
            elif txn.startswith("Sale"):
                trade_type = "SELL"
            else:
                continue
            trades.append({
                "trade_id": r[11] or _make_trade_id(r[5], ticker, trade_type, r[9]),
                "politician": r[5],
                "chamber": r[6],
                "party": r[7],
                "ticker": ticker,
                "issuer_name": r[1],
                "asset_type": asset_type,
                "trade_type": trade_type,
                "amount_range": r[4],
                "amount_midpoint": _parse_amount_midpoint(r[4]),
                "trade_date": (r[9] or "")[:10] or None,
                "published_date": (r[8] or "")[:10] or None,
                "source": "quiver",
            })
        except Exception as e:
            logger.debug("quiver row parse failed: %s", e)
    return trades


# ---------------- Capitol Trades (fallback; blocked by Vercel checkpoint) ----------------

def fetch_capitol_trades(pages: int = 3) -> list[dict]:
    from bs4 import BeautifulSoup  # noqa: F401  (kept for fallback parity)
    all_trades = []
    for page in range(1, pages + 1):
        try:
            url = f"{CAPITOL_BASE}/trades?page={page}"
            logger.info("Fetching %s", url)
            resp = _get(url)
            resp.raise_for_status()
            trades = _parse_capitol_page(resp.text)
            all_trades.extend(trades)
            if page < pages:
                time.sleep(2)
        except Exception as e:
            logger.error("capitoltrades page %d failed: %s", page, e)
            break
    return all_trades


def _parse_capitol_page(html: str) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    trades = []
    for row in soup.select("table tbody tr"):
        try:
            cells = row.find_all("td")
            if len(cells) < 8:
                continue
            politician = re.split(r"(Democrat|Republican|Other)\|",
                                  cells[0].get_text(strip=True))[0].strip()
            issuer_text = cells[1].get_text(strip=True)
            m = re.search(r"([A-Z]{1,5}(?:/[A-Z])?):US", issuer_text)
            ticker = m.group(1) if m else None
            if not ticker:
                continue
            trade_type_text, amount_range, price_text = "", "", ""
            for cell in cells[4:]:
                text = cell.get_text(strip=True)
                if text.upper().startswith(("BUY", "SELL")):
                    trade_type_text = text.upper().replace("*", "").strip()
                elif "K" in text.upper() and ("–" in text or "-" in text):
                    amount_range = text
                elif "$" in text:
                    price_text = text
            trade_type = ("BUY" if "BUY" in trade_type_text
                          else "SELL" if "SELL" in trade_type_text else None)
            if not trade_type:
                continue
            trade_date = cells[3].get_text(strip=True)
            trades.append({
                "trade_id": _make_trade_id(politician, ticker, trade_type, trade_date),
                "politician": politician,
                "ticker": ticker,
                "issuer_name": issuer_text.split(ticker)[0].strip(),
                "trade_type": trade_type,
                "amount_range": amount_range,
                "amount_midpoint": _parse_amount_midpoint(amount_range),
                "price": None if not price_text else float(
                    price_text.replace("$", "").replace(",", "")),
                "trade_date": trade_date,
                "published_date": cells[2].get_text(strip=True),
                "source": "capitoltrades",
            })
        except Exception as e:
            logger.debug("capitol row parse failed: %s", e)
    return trades


# ---------------- public API ----------------

def fetch_recent_trades(pages: int = 3, source: str = "auto") -> list[dict]:
    """Recent congress trades. source='auto' tries Quiver first (reliable),
    falls back to capitoltrades.com (usually blocked from datacenter IPs)."""
    if source in ("auto", "quiver"):
        try:
            trades = fetch_quiver_trades()
            if trades:
                logger.info("quiver: %d trades", len(trades))
                return trades
        except Exception as e:
            logger.warning("quiver fetch failed: %s", e)
        if source == "quiver":
            return []
    return fetch_capitol_trades(pages)


def congress_buy_signals(trades: list[dict], as_of: Optional[datetime] = None,
                         half_life_days: float = 45.0) -> dict[str, float]:
    """Aggregate recent purchases into per-ticker conviction scores in ~[-1, 1].

    Bigger disclosed amounts and more recent trades count more; several
    distinct politicians buying the same name counts more than one whale.
    """
    as_of = as_of or datetime.now(timezone.utc)
    raw: dict[str, float] = {}
    buyers: dict[str, set] = {}
    for t in trades:
        if t["trade_type"] != "BUY":
            continue
        tk = t["ticker"]
        dt = _parse_date(t.get("trade_date") or t.get("published_date") or "")
        age = max(0.0, (as_of - dt).days) if dt else 30.0
        recency = 0.5 ** (age / half_life_days)
        size = math.log10((t.get("amount_midpoint") or 10_000) / 1_000 + 1)
        raw[tk] = raw.get(tk, 0.0) + size * recency
        buyers.setdefault(tk, set()).add(t.get("politician", "?"))
    out = {}
    for tk, v in raw.items():
        diversity = 1 + 0.25 * min(len(buyers[tk]) - 1, 3)
        out[tk] = round(math.tanh(v * diversity / 3), 4)
    return out


def summarize_buys(trades: list[dict], limit: int = 5) -> list[dict]:
    """Most notable recent purchases, for the brief."""
    buys = [t for t in trades if t["trade_type"] == "BUY"]
    buys.sort(key=lambda t: (t.get("amount_midpoint") or 0), reverse=True)
    return buys[:limit]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = fetch_recent_trades()
    print(f"\n=== {len(results)} trades parsed ===")
    for t in results[:8]:
        print(json.dumps(t))
    sig = congress_buy_signals(results)
    top = sorted(sig.items(), key=lambda kv: -kv[1])[:10]
    print("\n=== top congress buy signals ===")
    for tk, s in top:
        print(f"  {tk:6s} {s:+.3f}")
