"""동적 universe + dual confirmation 의 mdd_cap × max_concurrent grid sweep.

목적: 동적 universe 가 약세장에서 BH 보다 나쁜 약점 (-20.59% vs BH -8.45%)
을 MDD cap 으로 해결할 수 있는지 검증 + 최적 max_concurrent 찾기.

Grid:
  mdd_cap        ∈ [None, 0.20, 0.25, 0.30, 0.35, 0.50]
  max_concurrent ∈ [5, 7, 10, 15]
  cooldown 10일 고정
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

MDD_GRID = [None, 0.20, 0.25, 0.30, 0.35, 0.50]
CONCURRENT_GRID = [5, 7, 10, 15]


def _calmar(ret, mdd, n_days):
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    return ((1 + ret) ** (1 / years) - 1) / abs(mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--enter-threshold", type=float, default=0.05)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--cooldown-days", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(
        symbols=tickers, cost_bps=args.cost_bps, max_pages=60,
    )

    print(f"=== Dynamic Dual Sweep ({len(tickers)} 종) ===")
    print(f"grid: mdd_cap={MDD_GRID}, max_concurrent={CONCURRENT_GRID}")
    print(f"= {len(MDD_GRID) * len(CONCURRENT_GRID)} 조합")
    print()

    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)

    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)
    bh_perf = analyze_regime_performance(bh.history, regimes)
    bh_calmar = _calmar(bh.overall["return"], bh.overall["max_drawdown"], bh.overall["n_days"])
    print(f"[BH baseline] final={bh.overall['final_value']:,.0f}원  ret={bh.overall['return']*100:+.2f}%  "
          f"MDD={bh.overall['max_drawdown']*100:+.2f}%  calmar={bh_calmar:+.2f}")
    print(f"             bull={bh_perf.get('bull',{}).get('cumulative_return',0)*100:+.2f}%  "
          f"bear={bh_perf.get('bear',{}).get('cumulative_return',0)*100:+.2f}%")
    print()

    rows = []
    print(f"  {'mdd':>6} {'conc':>5}  {'final':>14} {'ret':>9} {'MDD':>8} {'Calm':>6} "
          f"{'bull':>9} {'bear':>9}")
    for mdd in MDD_GRID:
        for nc in CONCURRENT_GRID:
            res = run_dual_dynamic_backtest(
                cfg, initial_cash=args.initial_cash,
                enter_threshold=args.enter_threshold,
                sizing_factor=200.0,
                cost_bps=args.cost_bps,
                max_concurrent=nc,
                mdd_cap=mdd,
                cooldown_days=args.cooldown_days,
            )
            o = res.overall
            calmar = _calmar(o["return"], o["max_drawdown"], o["n_days"])
            perf = analyze_regime_performance(res.history, regimes)
            bull = perf.get("bull", {}).get("cumulative_return", 0)
            bear = perf.get("bear", {}).get("cumulative_return", 0)

            mdd_label = "none" if mdd is None else f"{mdd:.0%}"
            print(f"  {mdd_label:>6} {nc:>5}  {o['final_value']:>14,.0f} "
                  f"{o['return']*100:>+8.2f}% {o['max_drawdown']*100:>+7.2f}% "
                  f"{calmar:>+5.2f} {bull*100:>+8.2f}% {bear*100:>+8.2f}%")
            rows.append({
                "mdd_cap": mdd, "max_concurrent": nc,
                "final": o["final_value"],
                "return": o["return"], "mdd": o["max_drawdown"],
                "calmar": calmar, "bull": bull, "bear": bear,
            })

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / f"sweep_dynamic_dual_{ts}.parquet"
    df.to_parquet(sweep_path)
    print(f"\n[saved] {sweep_path}")

    print()
    print("=== TOP 5 by Calmar ===")
    top = df.sort_values("calmar", ascending=False).head(5)
    for _, r in top.iterrows():
        mdd_l = "none" if r['mdd_cap'] is None or pd.isna(r['mdd_cap']) else f"{r['mdd_cap']:.0%}"
        print(f"  mdd={mdd_l:>5}  conc={int(r['max_concurrent'])}  "
              f"ret={r['return']*100:+7.2f}%  MDD={r['mdd']*100:+7.2f}%  calmar={r['calmar']:+.2f}  "
              f"bear={r['bear']*100:+7.2f}%")

    print()
    print("=== TOP 3 by bear (약세장 알파 회복) ===")
    top_bear = df.sort_values("bear", ascending=False).head(3)
    for _, r in top_bear.iterrows():
        mdd_l = "none" if r['mdd_cap'] is None or pd.isna(r['mdd_cap']) else f"{r['mdd_cap']:.0%}"
        print(f"  mdd={mdd_l:>5}  conc={int(r['max_concurrent'])}  "
              f"ret={r['return']*100:+7.2f}%  MDD={r['mdd']*100:+7.2f}%  bear={r['bear']*100:+7.2f}%")

    print()
    print(f"=== Best vs BH ===")
    best = df.sort_values("return", ascending=False).iloc[0]
    bh_ret = bh.overall["return"]
    mdd_l = "none" if best['mdd_cap'] is None or pd.isna(best['mdd_cap']) else f"{best['mdd_cap']:.0%}"
    print(f"  Best dynamic: mdd={mdd_l}, conc={int(best['max_concurrent'])}")
    print(f"    ret={best['return']*100:+.2f}%  vs BH {bh_ret*100:+.2f}%  → 차이 {(best['return']-bh_ret)*100:+.2f}%p")
    print(f"    bear={best['bear']*100:+.2f}%  vs BH bear={bh_perf.get('bear',{}).get('cumulative_return',0)*100:+.2f}%")


if __name__ == "__main__":
    main()
