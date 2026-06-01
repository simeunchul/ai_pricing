"""Risk Control 도구별 최적값 sweep + 비교.

세 도구를 isolation 으로 sweep 해서 각각 최적값 찾고, 최적값들끼리 비교 +
조합 시 추가 효과 측정.

도구:
  1. Trailing Stop (peak_close 대비 -X%)
  2. Portfolio MDD Cap (전체 자본 -X%, cooldown 10일)
  3. (선택) 두 도구 결합

Usage:
  python scripts/sweep_risk_controls.py
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
    ForeignFlowBacktestConfig,
    run_portfolio_backtest, run_buyhold_portfolio,
    universe_index, detect_regimes, analyze_regime_performance,
)
from autotrader.strategies.foreign_flow_proportional import ForeignFlowProportional


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_TICKERS = [
    "005930", "000660", "035420", "035720", "005380",
    "051910", "005490", "207940", "105560", "055550",
    "066570", "068270", "006400", "028260", "003670",
]

TRAILING_GRID = [None, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
MDD_CAP_GRID = [None, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def _calmar(total_return: float, mdd: float, n_days: int) -> float:
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    if years <= 0:
        return float("nan")
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized / abs(mdd)


def _run(cfg, sf, th, *, trailing=None, mdd=None, cooldown=10):
    strat = ForeignFlowProportional(
        sizing_factor=sf, min_threshold=th, max_position_per_symbol=10_000,
    )
    return run_portfolio_backtest(
        cfg, strategy=strat, initial_cash=10_000_000.0,
        trailing_stop_pct=trailing, mdd_cap=mdd, cooldown_days=cooldown,
    )


def _row(label, res, regimes):
    o = res.overall
    perf = analyze_regime_performance(res.history, regimes)
    bull = perf.get("bull", {}).get("cumulative_return", 0)
    bear = perf.get("bear", {}).get("cumulative_return", 0)
    return {
        "label": label,
        "final": o["final_value"],
        "return": o["return"],
        "mdd": o["max_drawdown"],
        "calmar": _calmar(o["return"], o["max_drawdown"], o["n_days"]),
        "bull_cum": bull,
        "bear_cum": bear,
    }


def _print_header():
    print(f"  {'설정':<22} {'final':>14} {'ret':>9} {'MDD':>8} "
          f"{'Calmar':>8} {'bull':>9} {'bear':>9}")


def _print_row(r):
    print(f"  {r['label']:<22} {r['final']:>14,.0f} "
          f"{r['return']*100:>+8.2f}% {r['mdd']*100:>+7.2f}% "
          f"{r['calmar']:>+7.2f} {r['bull_cum']*100:>+8.2f}% {r['bear_cum']*100:>+8.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--sizing-factor", type=float, default=200.0)
    ap.add_argument("--min-threshold", type=float, default=0.05)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--cooldown-days", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(
        symbols=tickers, cost_bps=args.cost_bps, max_pages=60,
    )

    print(f"=== Risk Control Sweep ({len(tickers)} symbols) ===")
    print(f"strategy: ForeignFlowProportional(sf={args.sizing_factor}, th={args.min_threshold})")
    print(f"cooldown: {args.cooldown_days} days")
    print()

    print("[Universe regime detection]")
    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)
    n_bull = (regimes == "bull").sum()
    n_bear = (regimes == "bear").sum()
    print(f"  bull {n_bull} / bear {n_bear} (total {len(regimes)})")
    print()

    print("[Buy & Hold baseline]")
    bh = run_buyhold_portfolio(cfg, initial_cash=10_000_000.0)
    bh_row = _row("BH baseline", bh, regimes)
    _print_header()
    _print_row(bh_row)
    print()

    # ===== Stage 1: Trailing Stop sweep =====
    print("[Stage 1] Trailing Stop sweep")
    _print_header()
    trailing_rows = []
    for tr in TRAILING_GRID:
        label = "trail=none" if tr is None else f"trail={tr:.2%}"
        res = _run(cfg, args.sizing_factor, args.min_threshold,
                   trailing=tr, mdd=None, cooldown=args.cooldown_days)
        r = _row(label, res, regimes)
        trailing_rows.append({**r, "trailing_stop_pct": tr})
        _print_row(r)
    print()

    # ===== Stage 2: MDD Cap sweep =====
    print("[Stage 2] MDD Cap sweep")
    _print_header()
    mdd_rows = []
    for cap in MDD_CAP_GRID:
        label = "mdd=none" if cap is None else f"mdd={cap:.2%}"
        res = _run(cfg, args.sizing_factor, args.min_threshold,
                   trailing=None, mdd=cap, cooldown=args.cooldown_days)
        r = _row(label, res, regimes)
        mdd_rows.append({**r, "mdd_cap": cap})
        _print_row(r)
    print()

    # ===== Pick best of each by Calmar =====
    df_trail = pd.DataFrame(trailing_rows)
    df_mdd = pd.DataFrame(mdd_rows)

    best_trail = df_trail.dropna(subset=["calmar"]).sort_values("calmar", ascending=False).iloc[0]
    best_mdd = df_mdd.dropna(subset=["calmar"]).sort_values("calmar", ascending=False).iloc[0]

    print("=== Best by Calmar ===")
    print(f"  best trailing : {best_trail['label']}  "
          f"(ret={best_trail['return']*100:+.2f}%  MDD={best_trail['mdd']*100:+.2f}%  "
          f"calmar={best_trail['calmar']:+.2f})")
    print(f"  best MDD cap  : {best_mdd['label']}  "
          f"(ret={best_mdd['return']*100:+.2f}%  MDD={best_mdd['mdd']*100:+.2f}%  "
          f"calmar={best_mdd['calmar']:+.2f})")
    print()

    # ===== Stage 3: Combined (best trailing + best mdd) =====
    print("[Stage 3] Combined (best trailing + best MDD cap)")
    _print_header()
    combined_res = _run(
        cfg, args.sizing_factor, args.min_threshold,
        trailing=best_trail["trailing_stop_pct"] if not pd.isna(best_trail["trailing_stop_pct"]) else None,
        mdd=best_mdd["mdd_cap"] if not pd.isna(best_mdd["mdd_cap"]) else None,
        cooldown=args.cooldown_days,
    )
    combined_r = _row("trail+mdd combined", combined_res, regimes)
    _print_row(combined_r)
    print()

    # ===== Final summary table =====
    print("=" * 90)
    print("=== FINAL COMPARISON ===")
    print("=" * 90)
    _print_header()
    _print_row(bh_row)
    # vanilla
    res_van = _run(cfg, args.sizing_factor, args.min_threshold,
                    trailing=None, mdd=None)
    _print_row(_row("vanilla (no risk)", res_van, regimes))
    _print_row({**best_trail.to_dict(), "label": f"BEST trail ({best_trail['label']})"})
    _print_row({**best_mdd.to_dict(), "label": f"BEST mdd ({best_mdd['label']})"})
    _print_row({**combined_r, "label": "COMBINED (trail+mdd)"})

    # save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "tickers": tickers,
            "sizing_factor": args.sizing_factor,
            "min_threshold": args.min_threshold,
            "cooldown_days": args.cooldown_days,
        },
        "buyhold": bh_row,
        "trailing_sweep": trailing_rows,
        "mdd_sweep": mdd_rows,
        "best_trailing": best_trail.to_dict(),
        "best_mdd_cap": best_mdd.to_dict(),
        "combined": combined_r,
    }
    summary_path = out_dir / f"sweep_risk_controls_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    main()
