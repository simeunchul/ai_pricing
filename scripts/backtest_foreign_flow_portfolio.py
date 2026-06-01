"""ForeignFlowProportional 전략 — Portfolio-level backtest.

Trade-level metric (mean / Sharpe per trade) 의 misleading 함을 회피하기 위해
실제 자본 흐름 (cash + position × close) 을 매일 마크해서 final portfolio value
를 매수후보유 baseline 과 비교한다.

종목당 동일 자본 (initial_cash / n_symbols) 으로 시작하여 독립 운용.
매수: cash 한도 + sizing_factor × |flow_ratio| 의 작은 값.
매도: 보유분 한도 + sizing_factor × |flow_ratio| 의 작은 값 (부분매도 지원).

Usage:
  단일 케이스    python scripts/backtest_foreign_flow_portfolio.py
  custom params  python scripts/backtest_foreign_flow_portfolio.py --sizing-factor 100 --min-threshold 0.02
  sweep           python scripts/backtest_foreign_flow_portfolio.py --sweep
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
)
from autotrader.strategies.foreign_flow_proportional import ForeignFlowProportional


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_TICKERS = [
    "005930", "000660", "035420", "005380", "051910",
]

SWEEP_SIZING_FACTORS = [50, 100, 200, 500]
SWEEP_MIN_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]


def _print_summary(label: str, result, baseline_overall: dict | None = None):
    print(f"\n=== {label} ===")
    print(f"  initial cash    : {result.overall['initial_cash']:>14,.0f} 원")
    print(f"  final value     : {result.overall['final_value']:>14,.0f} 원")
    print(f"  total return    : {result.overall['return']*100:>+14.2f} %")
    print(f"  max drawdown    : {result.overall['max_drawdown']*100:>+14.2f} %")
    print(f"  trading days    : {result.overall['n_days']}")

    if baseline_overall is not None:
        excess = result.overall["return"] - baseline_overall["return"]
        print(f"  vs buy-hold     : {excess*100:>+14.2f} %p")

    if not result.summary.empty:
        print()
        print(f"  {'symbol':<8} {'start':>12} {'final':>14} {'return':>10} {'mdd':>10}")
        for _, r in result.summary.iterrows():
            print(f"  {r['symbol']:<8} {r['start_value']:>12,.0f} "
                  f"{r['final_value']:>14,.0f} "
                  f"{r['return']*100:>+9.2f}% {r['max_drawdown']*100:>+9.2f}%")


def _run_one(cfg: ForeignFlowBacktestConfig, sf: float, th: float, cash: float):
    strat = ForeignFlowProportional(
        sizing_factor=sf, min_threshold=th, max_position_per_symbol=10_000,
    )
    return run_portfolio_backtest(cfg, strategy=strat, initial_cash=cash)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--sizing-factor", type=float, default=50.0)
    ap.add_argument("--min-threshold", type=float, default=0.02)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="grid sweep 모드 (sf × th)")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(
        symbols=tickers,
        cost_bps=args.cost_bps,
        max_pages=args.max_pages,
        start=args.start,
        end=args.end,
    )

    print("=== Portfolio-level Backtest ===")
    print(f"tickers       : {len(tickers)} ({', '.join(tickers)})")
    print(f"initial cash  : {args.initial_cash:,.0f} 원")
    print(f"per-symbol    : {args.initial_cash/len(tickers):,.0f} 원")
    print(f"cost (round)  : {args.cost_bps} bps")
    print()

    # Buy-hold baseline (한 번만 계산)
    print("[1/2] Buy & Hold baseline 시뮬...")
    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)
    _print_summary("매수후보유 (Buy & Hold)", bh)

    if args.sweep:
        print()
        print("[2/2] Strategy grid sweep...")
        rows = []
        best = None
        for sf in SWEEP_SIZING_FACTORS:
            for th in SWEEP_MIN_THRESHOLDS:
                res = _run_one(cfg, sf, th, args.initial_cash)
                excess = res.overall["return"] - bh.overall["return"]
                rows.append({
                    "sizing_factor": sf,
                    "min_threshold": th,
                    "final_value": res.overall["final_value"],
                    "return": res.overall["return"],
                    "max_drawdown": res.overall["max_drawdown"],
                    "vs_buyhold": excess,
                })
                print(f"  sf={sf:>5.0f}  th={th:.3f}  "
                      f"final={res.overall['final_value']:>13,.0f}원  "
                      f"ret={res.overall['return']*100:>+7.2f}%  "
                      f"mdd={res.overall['max_drawdown']*100:>+7.2f}%  "
                      f"vs_BH={excess*100:>+6.2f}%p")
                if best is None or excess > best[2]:
                    best = (sf, th, excess, res)

        df = pd.DataFrame(rows)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = out_dir / f"sweep_portfolio_{ts}.parquet"
        df.to_parquet(sweep_path)
        print(f"\n[saved] sweep grid : {sweep_path}")

        print()
        print("=== TOP 3 by vs_buyhold (excess) ===")
        top = df.sort_values("vs_buyhold", ascending=False).head(3)
        for _, r in top.iterrows():
            print(f"  sf={r['sizing_factor']:>5.0f}  th={r['min_threshold']:.3f}  "
                  f"final={r['final_value']:>13,.0f}원  "
                  f"vs_BH={r['vs_buyhold']*100:>+6.2f}%p")

        print()
        if best is not None:
            sf, th, excess, res = best
            _print_summary(f"BEST 전략 (sf={sf}, th={th})", res, baseline_overall=bh.overall)
    else:
        print()
        print(f"[2/2] Strategy (sf={args.sizing_factor}, th={args.min_threshold})...")
        res = _run_one(cfg, args.sizing_factor, args.min_threshold, args.initial_cash)
        _print_summary(
            f"외국인 추종 (sf={args.sizing_factor}, th={args.min_threshold})",
            res, baseline_overall=bh.overall,
        )

        # save history
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        hist_path = out_dir / f"portfolio_history_{ts}.parquet"
        res.history.to_parquet(hist_path)
        bh_hist_path = out_dir / f"portfolio_buyhold_history_{ts}.parquet"
        bh.history.to_parquet(bh_hist_path)
        print(f"\n[saved] strategy history : {hist_path}")
        print(f"[saved] buyhold history  : {bh_hist_path}")


if __name__ == "__main__":
    main()
