"""실시장 IV surface 수집 — yfinance 로 SPY 옵션 chain.

플랜은 KOSPI200 인데 pykrx 의 KRX 옵션 API 가 인증 차단. SPY (S&P500 ETF)
옵션은 CBOE 가 무료 무인증 제공 → yfinance 통해 받음.
B2 Deep Calib 에 투입할 5×5 IV grid 추출.

Output:
  data/market_snapshots/spy_iv_<YYYYMMDD>.json
  - spot, r, snapshot_time
  - moneyness × maturity 로 5×5 IV grid

Usage:
  python scripts/fetch_spy_iv.py
  python scripts/fetch_spy_iv.py --ticker SPY --moneyness 0.85,0.93,1.0,1.07,1.15
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import yfinance as yf


def fetch_iv_surface(ticker: str = "SPY",
                     target_moneyness: list[float] | None = None,
                     target_T_months: list[float] | None = None,
                     r: float = 0.045) -> dict:
    """Fetch and bilinearly-interpolate option IVs onto a regular grid."""
    if target_moneyness is None:
        target_moneyness = [0.90, 0.95, 1.00, 1.05, 1.10]
    if target_T_months is None:
        target_T_months = [1.0, 3.0, 6.0, 12.0, 18.0]   # months

    tk = yf.Ticker(ticker)
    hist = tk.history(period="2d")
    spot = float(hist["Close"].iloc[-1])
    snap_dt = datetime.datetime.now()

    expiries = tk.options
    print(f"[fetch] {ticker} spot={spot:.2f}, {len(expiries)} expirations")

    # Compute T (years) for each expiry, gather all (T, K, IV) points
    pts = []
    for exp_str in expiries:
        exp = datetime.datetime.strptime(exp_str, "%Y-%m-%d")
        T = (exp - snap_dt).total_seconds() / (365.25 * 86400)
        if T <= 1e-3 or T > 3.0:
            continue
        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue
        # only ATM-ish strikes (filter heavy wings to reduce noise)
        calls = chain.calls
        calls = calls[(calls["strike"] > 0.5 * spot) & (calls["strike"] < 1.5 * spot)]
        for _, row in calls.iterrows():
            iv = row["impliedVolatility"]
            if iv is None or np.isnan(iv) or iv <= 0.01 or iv > 3.0:
                continue
            pts.append((T, row["strike"] / spot, float(iv)))

    pts = np.array(pts)
    print(f"[fetch] collected {len(pts)} (T, m, IV) points")

    # Build target grid
    grid = np.full((len(target_T_months), len(target_moneyness)), np.nan)
    for i, T_m in enumerate(target_T_months):
        T_yr = T_m / 12
        # nearest available T (within 25% tolerance)
        T_arr = pts[:, 0]
        T_diff = np.abs(T_arr - T_yr)
        T_close_mask = T_diff < max(T_yr * 0.25, 1 / 365)
        sub = pts[T_close_mask]
        if len(sub) < 3:
            continue
        for j, m in enumerate(target_moneyness):
            # nearest moneyness within 5%
            m_diff = np.abs(sub[:, 1] - m)
            best_idx = np.argsort(m_diff)[:3]    # 3 nearest, average
            top3 = sub[best_idx]
            if m_diff[best_idx[0]] < 0.05:
                # weighted by inverse moneyness distance
                w = 1.0 / (np.abs(top3[:, 1] - m) + 1e-3)
                w /= w.sum()
                grid[i, j] = float((top3[:, 2] * w).sum())

    return {
        "ticker": ticker,
        "spot": spot,
        "snapshot_time": snap_dt.isoformat(),
        "r": r,
        "moneyness": target_moneyness,
        "maturities_months": target_T_months,
        "iv_grid": grid.tolist(),
        "n_pts_collected": len(pts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--r", type=float, default=0.045)
    ap.add_argument("--out-dir", default="data/market_snapshots")
    args = ap.parse_args()

    res = fetch_iv_surface(args.ticker, r=args.r)

    grid = np.array(res["iv_grid"])
    print(f"\n=== IV surface ({res['ticker']}, spot={res['spot']:.2f}) ===")
    print(f"{'T(M)\\M':>8s} | " + " ".join(f"{m:>6.2f}" for m in res["moneyness"]))
    print("-" * 60)
    for i, T_m in enumerate(res["maturities_months"]):
        row = grid[i]
        cells = [f"{v:>6.3f}" if not np.isnan(v) else "  nan " for v in row]
        print(f"{T_m:>6.0f}M  | " + " ".join(cells))

    n_valid = int((~np.isnan(grid)).sum())
    print(f"\nValid cells: {n_valid}/{grid.size}")

    today = datetime.datetime.now().strftime("%Y%m%d")
    out = Path(args.out_dir) / f"{args.ticker.lower()}_iv_{today}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
