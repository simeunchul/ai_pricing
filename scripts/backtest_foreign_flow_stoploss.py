"""Stop Loss sweep — Portfolio-level backtest.

이전 portfolio backtest 의 best params (sf=200, th=0.05) 고정 + stop_loss_pct 만
스윕해서 MDD 와 final return 의 trade-off 측정.

Calmar ratio = 연환산 수익률 / |MDD| 로 위험조정 수익률 비교.

Usage:
  python scripts/backtest_foreign_flow_stoploss.py
  python scripts/backtest_foreign_flow_stoploss.py --sizing-factor 100 --min-threshold 0.02
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

DEFAULT_STOP_LOSSES = [None, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]


def _calmar(total_return: float, mdd: float, n_days: int) -> float:
    """Calmar ratio = annualized return / |MDD|.

    annualized = (1 + total_return) ** (252 / n_days) - 1
    """
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    if years <= 0:
        return float("nan")
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized / abs(mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--sizing-factor", type=float, default=200.0)
    ap.add_argument("--min-threshold", type=float, default=0.05)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--stop-losses",
                    default=",".join(["none" if s is None else str(s)
                                      for s in DEFAULT_STOP_LOSSES]),
                    help="콤마 구분 stop_loss 비율 ('none' = 비활성)")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    sl_strs = [s.strip() for s in args.stop_losses.split(",") if s.strip()]
    stop_losses = [None if s.lower() == "none" else float(s) for s in sl_strs]

    cfg = ForeignFlowBacktestConfig(
        symbols=tickers,
        cost_bps=args.cost_bps,
        max_pages=args.max_pages,
    )

    print("=== Stop Loss Sweep — Portfolio Level ===")
    print(f"tickers       : {len(tickers)} ({', '.join(tickers)})")
    print(f"sizing_factor : {args.sizing_factor}")
    print(f"min_threshold : {args.min_threshold}")
    print(f"initial cash  : {args.initial_cash:,.0f} 원")
    print(f"stop_losses   : {stop_losses}")
    print()

    # Buy-hold baseline
    print("[1/2] Buy & Hold baseline...")
    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)
    bh_calmar = _calmar(bh.overall["return"], bh.overall["max_drawdown"], bh.overall["n_days"])
    print(f"  BH: final={bh.overall['final_value']:>12,.0f}원  "
          f"ret={bh.overall['return']*100:>+7.2f}%  "
          f"mdd={bh.overall['max_drawdown']*100:>+7.2f}%  "
          f"calmar={bh_calmar:>+5.2f}")
    print()

    # Sweep
    print("[2/2] Stop loss sweep...")
    rows = []
    for sl in stop_losses:
        strat = ForeignFlowProportional(
            sizing_factor=args.sizing_factor,
            min_threshold=args.min_threshold,
            max_position_per_symbol=10_000,
        )
        res = run_portfolio_backtest(cfg, strategy=strat,
                                      initial_cash=args.initial_cash,
                                      stop_loss_pct=sl)
        ret = res.overall["return"]
        mdd = res.overall["max_drawdown"]
        calmar = _calmar(ret, mdd, res.overall["n_days"])
        excess_vs_bh = ret - bh.overall["return"]
        rows.append({
            "stop_loss_pct": sl,
            "final_value": res.overall["final_value"],
            "return": ret,
            "max_drawdown": mdd,
            "calmar": calmar,
            "vs_buyhold": excess_vs_bh,
        })

        sl_label = "none" if sl is None else f"{sl:.2%}"
        print(f"  sl={sl_label:>8}  final={res.overall['final_value']:>12,.0f}원  "
              f"ret={ret*100:>+7.2f}%  "
              f"mdd={mdd*100:>+7.2f}%  "
              f"calmar={calmar:>+5.2f}  "
              f"vs_BH={excess_vs_bh*100:>+6.2f}%p")

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / f"sweep_stoploss_{ts}.parquet"
    df.to_parquet(sweep_path)
    print(f"\n[saved] sweep grid : {sweep_path}")

    print()
    print("=== TOP 3 by Calmar (위험조정 수익률) ===")
    valid = df.dropna(subset=["calmar"]).sort_values("calmar", ascending=False).head(3)
    for _, r in valid.iterrows():
        sl = r["stop_loss_pct"]
        sl_label = "none" if sl is None or pd.isna(sl) else f"{sl:.2%}"
        print(f"  sl={sl_label:>8}  ret={r['return']*100:>+7.2f}%  "
              f"mdd={r['max_drawdown']*100:>+7.2f}%  calmar={r['calmar']:>+5.2f}")

    print()
    print("=== TOP 3 by lowest MDD (안전성) ===")
    by_mdd = df.sort_values("max_drawdown", ascending=False).head(3)
    for _, r in by_mdd.iterrows():
        sl = r["stop_loss_pct"]
        sl_label = "none" if sl is None or pd.isna(sl) else f"{sl:.2%}"
        print(f"  sl={sl_label:>8}  ret={r['return']*100:>+7.2f}%  "
              f"mdd={r['max_drawdown']*100:>+7.2f}%  calmar={r['calmar']:>+5.2f}")


if __name__ == "__main__":
    main()
