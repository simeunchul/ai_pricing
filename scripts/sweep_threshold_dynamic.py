"""Threshold sweep on dynamic universe + dual confirmation backtest.

목적: enter_threshold 완화 (0.05 → 0.03 등) 시 진짜 알파 유지되는지 검증.
백테스트 안 된 영역으로 가기 전 sanity check.

Grid:
  enter_threshold ∈ [0.02, 0.03, 0.04, 0.05, 0.06]
  max_concurrent = 7 고정 (이전 sweep best)
  mdd_cap = 30% / cooldown 10일 고정
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

import pandas as pd

from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig, run_buyhold_portfolio, run_dual_dynamic_backtest,
    universe_index, detect_regimes, analyze_regime_performance,
)


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_TICKERS = [
    "005930", "000660", "035420", "035720", "005380",
    "051910", "005490", "207940", "105560", "055550",
    "066570", "068270", "006400", "028260", "003670",
    "000270", "012330", "010130", "011170", "086790",
    "024110", "003490", "042660", "042700", "034020",
    "009830", "011200", "011780", "086280", "047810",
    "015760", "030200", "003550", "004020", "047040",
    "138930", "005935", "326030", "395400", "000720",
]

THRESHOLD_GRID = [0.02, 0.03, 0.04, 0.05, 0.06]


def _calmar(ret, mdd, n_days):
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    return ((1 + ret) ** (1 / years) - 1) / abs(mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=7)
    ap.add_argument("--mdd-cap", type=float, default=0.30)
    ap.add_argument("--cooldown-days", type=int, default=10)
    args = ap.parse_args()

    cfg = ForeignFlowBacktestConfig(
        symbols=DEFAULT_TICKERS, cost_bps=25.0, max_pages=60,
    )

    print(f"=== Threshold Sweep — Dynamic Universe ({len(DEFAULT_TICKERS)} 종) ===")
    print(f"max_concurrent={args.max_concurrent}, mdd_cap={args.mdd_cap}")
    print()

    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)

    bh = run_buyhold_portfolio(cfg, initial_cash=10_000_000.0)
    bh_perf = analyze_regime_performance(bh.history, regimes)
    bh_calmar = _calmar(bh.overall["return"], bh.overall["max_drawdown"], bh.overall["n_days"])
    print(f"[BH baseline] ret={bh.overall['return']*100:+.2f}%  "
          f"MDD={bh.overall['max_drawdown']*100:+.2f}%  calmar={bh_calmar:+.2f}  "
          f"bull={bh_perf.get('bull',{}).get('cumulative_return',0)*100:+.2f}% "
          f"bear={bh_perf.get('bear',{}).get('cumulative_return',0)*100:+.2f}%")
    print()

    rows = []
    print(f"  {'th':>5}  {'final':>14} {'ret':>9} {'MDD':>8} {'Calm':>6} "
          f"{'bull':>9} {'bear':>9} {'avg_pos':>8}")
    for th in THRESHOLD_GRID:
        res = run_dual_dynamic_backtest(
            cfg, initial_cash=10_000_000.0,
            enter_threshold=th, sizing_factor=200.0, cost_bps=25.0,
            max_concurrent=args.max_concurrent,
            mdd_cap=args.mdd_cap, cooldown_days=args.cooldown_days,
        )
        o = res.overall
        calmar = _calmar(o["return"], o["max_drawdown"], o["n_days"])
        perf = analyze_regime_performance(res.history, regimes)
        bull = perf.get("bull", {}).get("cumulative_return", 0)
        bear = perf.get("bear", {}).get("cumulative_return", 0)
        avg_pos = res.history["position"].mean()

        print(f"  {th:>5.3f}  {o['final_value']:>14,.0f} "
              f"{o['return']*100:>+8.2f}% {o['max_drawdown']*100:>+7.2f}% "
              f"{calmar:>+5.2f} {bull*100:>+8.2f}% {bear*100:>+8.2f}% {avg_pos:>7.2f}")
        rows.append({
            "threshold": th,
            "final": o["final_value"],
            "return": o["return"], "mdd": o["max_drawdown"],
            "calmar": calmar, "bull": bull, "bear": bear,
            "avg_position_count": avg_pos,
        })

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "data" / f"sweep_threshold_{ts}.parquet"
    df.to_parquet(out_path)
    print(f"\n[saved] {out_path}")

    print()
    print("=== TOP by Calmar ===")
    for _, r in df.sort_values("calmar", ascending=False).iterrows():
        print(f"  th={r['threshold']:.3f}  ret={r['return']*100:+.2f}%  "
              f"MDD={r['mdd']*100:+.2f}%  calmar={r['calmar']:+.2f}  "
              f"avg_pos={r['avg_position_count']:.2f}")


if __name__ == "__main__":
    main()
