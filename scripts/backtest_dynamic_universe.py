"""동적 universe + Dual Confirmation 백테스트.

영웅문 사용 방식 시뮬레이션:
  매일 universe 종목 스캔 → 어제 외인+기관 둘 다 +5%↑ 종목 매수
  보유 종목 중 어제 둘 다 -5%↓ 종목 매도
  최대 max_concurrent (default 10) 종목 동시 보유

비교:
  A. Buy & Hold (universe 균등 매수후보유)
  B. 고정 universe + Dual (이전 best, n=15 sample)
  C. 동적 universe + Dual (이번 mode, 자동 발견)

Usage:
  python scripts/backtest_dynamic_universe.py
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
    run_buyhold_portfolio, run_portfolio_backtest, run_dual_dynamic_backtest,
    universe_index, detect_regimes, analyze_regime_performance,
)
from autotrader.strategies.foreign_inst_flow import ForeignInstFlowFollow


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 40 종 universe (기존 15 + 추가 25)
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


def _calmar(total_return: float, mdd: float, n_days: int) -> float:
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized / abs(mdd)


def _print_row(label, overall, regimes_perf=None):
    calmar = _calmar(overall["return"], overall["max_drawdown"], overall["n_days"])
    s = (f"  {label:<35} {overall['final_value']:>14,.0f} "
         f"{overall['return']*100:>+8.2f}% {overall['max_drawdown']*100:>+7.2f}% "
         f"{calmar:>+7.2f}")
    if regimes_perf:
        bull = regimes_perf.get("bull", {}).get("cumulative_return", 0)
        bear = regimes_perf.get("bear", {}).get("cumulative_return", 0)
        s += f" {bull*100:>+8.2f}% {bear*100:>+8.2f}%"
    print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--enter-threshold", type=float, default=0.05)
    ap.add_argument("--sizing-factor", type=float, default=200.0)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--max-concurrent", type=int, default=10,
                    help="동적 모드에서 동시 보유 가능 최대 종목 수")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(
        symbols=tickers, cost_bps=args.cost_bps, max_pages=60,
    )

    print(f"=== Dynamic Universe + Dual Confirmation ({len(tickers)} 종) ===")
    print(f"enter_threshold={args.enter_threshold}, max_concurrent={args.max_concurrent}")
    print(f"initial cash    = {args.initial_cash:,.0f} 원")
    print()

    print("[Universe regime]")
    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)
    print(f"  bull {(regimes=='bull').sum()}일 / bear {(regimes=='bear').sum()}일")
    print()

    print("=" * 110)
    print(f"  {'전략':<35} {'final':>14} {'ret':>9} {'MDD':>8} {'Calmar':>8} {'bull':>9} {'bear':>9}")
    print("=" * 110)

    # A. Buy & Hold
    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)
    bh_perf = analyze_regime_performance(bh.history, regimes)
    _print_row("A. Buy & Hold (40종 균등)", bh.overall, bh_perf)

    # B. 고정 universe + Dual + MDD cap 25%
    strat_b = ForeignInstFlowFollow(
        enter_threshold=args.enter_threshold, sizing_factor=args.sizing_factor,
        min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res_b = run_portfolio_backtest(
        cfg, strategy=strat_b, initial_cash=args.initial_cash,
        mdd_cap=0.25, cooldown_days=10,
        signal_keys=("flow_ratio", "inst_ratio"),
    )
    perf_b = analyze_regime_performance(res_b.history, regimes)
    _print_row("B. 고정 + Dual + MDD25% (이전 best)", res_b.overall, perf_b)

    # C. 동적 universe + Dual
    res_c = run_dual_dynamic_backtest(
        cfg, initial_cash=args.initial_cash,
        enter_threshold=args.enter_threshold,
        sizing_factor=args.sizing_factor,
        cost_bps=args.cost_bps,
        max_concurrent=args.max_concurrent,
    )
    # res_c.history 는 _TOTAL_ 행만 있음 — regime 분석 변환
    hist_c = res_c.history.copy()
    hist_c = hist_c.rename(columns={"value": "value"})
    perf_c = analyze_regime_performance(hist_c, regimes)
    _print_row(f"C. 동적 universe + Dual (max {args.max_concurrent})",
               res_c.overall, perf_c)

    print()

    # 진입 빈도 측정 (동적 universe)
    print(f"[동적 universe 진입 통계]")
    days_with_holding = (hist_c[hist_c["symbol"] == "_TOTAL_"]["position"] > 0).sum()
    avg_holdings = hist_c[hist_c["symbol"] == "_TOTAL_"]["position"].mean()
    print(f"  보유 종목 0 초과 일수: {days_with_holding} / {len(hist_c)}일")
    print(f"  평균 동시 보유 종목 수: {avg_holdings:.2f}")
    print()

    # save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "tickers": tickers, "enter_threshold": args.enter_threshold,
            "sizing_factor": args.sizing_factor, "max_concurrent": args.max_concurrent,
        },
        "results": {
            "buyhold": {"final": bh.overall["final_value"], "return": bh.overall["return"], "mdd": bh.overall["max_drawdown"]},
            "fixed_dual": {"final": res_b.overall["final_value"], "return": res_b.overall["return"], "mdd": res_b.overall["max_drawdown"]},
            "dynamic_dual": {"final": res_c.overall["final_value"], "return": res_c.overall["return"], "mdd": res_c.overall["max_drawdown"]},
        },
        "dynamic_holdings": {
            "days_with_holding": int(days_with_holding),
            "total_days": len(hist_c),
            "avg_concurrent": float(avg_holdings),
        },
    }
    summary_path = out_dir / f"dynamic_dual_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
