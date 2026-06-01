"""ForeignFlowProportional 전략의 sizing_factor / min_threshold grid sweep.

목적: 사용자 의도 (외국인 매매 비율에 비례한 포지션 사이징) 의 최적 파라미터를
brute-force 탐색.

평가 지표:
  - mean_return        거래비용 후 거래당 평균 수익률 (양수면 알파)
  - sharpe_per_trade   거래당 Sharpe (변동성 대비)
  - total_return_sum   누적 수익률 합 (5종 단순 합)
  - n_trades           샘플 수 (통계적 유의성)

매수후보유 비교는 의도적으로 제외 (불장 편향).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

import pandas as pd

from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig, run_foreign_flow_backtest,
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

DEFAULT_SIZING_FACTORS = [50, 100, 200, 300, 500, 1000]
DEFAULT_MIN_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]


def _evaluate(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n": 0, "mean": float("nan"), "std": float("nan"),
            "sharpe": float("nan"), "total": 0.0, "hit": float("nan"),
            "avg_hold": float("nan"),
        }
    n = len(trades)
    mean = float(trades["net_return"].mean())
    std = float(trades["net_return"].std())
    sharpe = mean / std if std > 0 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "total": float(trades["net_return"].sum()),
        "hit": float((trades["net_return"] > 0).mean()),
        "avg_hold": float(trades["hold_days"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--sizing-factors", default=",".join(str(s) for s in DEFAULT_SIZING_FACTORS))
    ap.add_argument("--min-thresholds", default=",".join(str(t) for t in DEFAULT_MIN_THRESHOLDS))
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    sizing_factors = [float(s) for s in args.sizing_factors.split(",")]
    min_thresholds = [float(t) for t in args.min_thresholds.split(",")]

    print("=== Foreign Flow Proportional — Grid Sweep ===")
    print(f"tickers          : {len(tickers)} ({', '.join(tickers)})")
    print(f"sizing_factors   : {sizing_factors}")
    print(f"min_thresholds   : {min_thresholds}")
    print(f"grid size        : {len(sizing_factors) * len(min_thresholds)} combinations")
    print()

    rows = []
    for sf in sizing_factors:
        for th in min_thresholds:
            cfg = ForeignFlowBacktestConfig(
                symbols=tickers,
                cost_bps=args.cost_bps,
                max_pages=args.max_pages,
                start=args.start,
                end=args.end,
            )
            strat = ForeignFlowProportional(
                sizing_factor=sf,
                min_threshold=th,
                max_position_per_symbol=10_000,    # 사실상 비활성
            )
            trades = run_foreign_flow_backtest(cfg, strategy=strat)
            metrics = _evaluate(trades)

            row = {
                "sizing_factor": sf,
                "min_threshold": th,
                **metrics,
            }
            rows.append(row)

            print(f"  sf={sf:>5.0f}  th={th:.3f}  "
                  f"n={metrics['n']:>4}  hit={metrics['hit']*100:>5.1f}%  "
                  f"mean={metrics['mean']*100:>+7.4f}%  "
                  f"sharpe={metrics['sharpe']:>+7.4f}  "
                  f"total={metrics['total']*100:>+8.2f}%  "
                  f"hold={metrics['avg_hold']:>5.1f}d")

    df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / f"sweep_foreign_flow_prop_{ts}.parquet"
    df.to_parquet(sweep_path)
    print(f"\n[saved] sweep grid : {sweep_path}")

    # Top by sharpe
    print()
    print("=== TOP 5 by Sharpe (per-trade) — n>=50 ===")
    sub = df[df["n"] >= 50].copy()
    if not sub.empty:
        top_sharpe = sub.sort_values("sharpe", ascending=False).head(5)
        for _, r in top_sharpe.iterrows():
            print(f"  sf={r['sizing_factor']:>5.0f}  th={r['min_threshold']:.3f}  "
                  f"n={int(r['n']):>4}  mean={r['mean']*100:>+7.4f}%  "
                  f"sharpe={r['sharpe']:>+7.4f}  total={r['total']*100:>+7.2f}%")

    print()
    print("=== TOP 5 by total_return — n>=50 ===")
    if not sub.empty:
        top_total = sub.sort_values("total", ascending=False).head(5)
        for _, r in top_total.iterrows():
            print(f"  sf={r['sizing_factor']:>5.0f}  th={r['min_threshold']:.3f}  "
                  f"n={int(r['n']):>4}  mean={r['mean']*100:>+7.4f}%  "
                  f"sharpe={r['sharpe']:>+7.4f}  total={r['total']*100:>+7.2f}%")

    print()
    print("=== TOP 5 by mean_return (per-trade) — n>=50 ===")
    if not sub.empty:
        top_mean = sub.sort_values("mean", ascending=False).head(5)
        for _, r in top_mean.iterrows():
            print(f"  sf={r['sizing_factor']:>5.0f}  th={r['min_threshold']:.3f}  "
                  f"n={int(r['n']):>4}  mean={r['mean']*100:>+7.4f}%  "
                  f"sharpe={r['sharpe']:>+7.4f}  total={r['total']*100:>+7.2f}%")


if __name__ == "__main__":
    main()
