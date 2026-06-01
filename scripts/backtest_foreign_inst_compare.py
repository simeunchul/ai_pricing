"""외국인 vs 외국인+기관 동방향 신호 비교 backtest.

같은 universe / 자본 / 기간에서:
  1. ForeignFlowProportional        — 외국인만 추종
  2. ForeignInstFlowFollow          — 외국인 + 기관 동방향만
  + 각각 MDD cap 25% 적용 비교

평가:
  - Final value, return, MDD, Calmar
  - Bull/Bear 구간 분리 cumulative return
  - 거래 수 (외국인+기관 동방향이라 더 적을 것으로 예상)

Usage:
  python scripts/backtest_foreign_inst_compare.py
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
from autotrader.strategies.foreign_inst_flow import ForeignInstFlowFollow


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


def _calmar(total_return: float, mdd: float, n_days: int) -> float:
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    if years <= 0:
        return float("nan")
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized / abs(mdd)


def _print_row(label, res, regimes):
    o = res.overall
    perf = analyze_regime_performance(res.history, regimes)
    bull = perf.get("bull", {}).get("cumulative_return", 0)
    bear = perf.get("bear", {}).get("cumulative_return", 0)
    calmar = _calmar(o["return"], o["max_drawdown"], o["n_days"])

    # 매수/매도 거래 횟수 — history 의 cash 변화로 근사
    hist = res.history.sort_values(["symbol", "date"])
    n_buys = 0
    n_sells = 0
    for sym, sub in hist.groupby("symbol"):
        cash_diff = sub["cash"].diff()
        n_buys += (cash_diff < -1).sum()    # cash 줄어듬 = 매수
        n_sells += (cash_diff > 1).sum()    # cash 늘어남 = 매도

    print(f"  {label:<32} {o['final_value']:>13,.0f} {o['return']*100:>+8.2f}% "
          f"{o['max_drawdown']*100:>+7.2f}% {calmar:>+6.2f} "
          f"{bull*100:>+7.2f}% {bear*100:>+7.2f}% "
          f"{n_buys:>5d}/{n_sells:>5d}")


def _print_header():
    print(f"  {'전략':<32} {'final':>13} {'ret':>9} {'MDD':>8} {'Calm':>6} "
          f"{'bull':>8} {'bear':>8} {'buys/sells':>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--enter-threshold", type=float, default=0.05)
    ap.add_argument("--sizing-factor", type=float, default=200.0)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--mdd-cap", type=float, default=0.25)
    ap.add_argument("--cooldown-days", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(
        symbols=tickers, cost_bps=args.cost_bps, max_pages=60,
    )

    print(f"=== 외국인 vs 외국인+기관 동방향 비교 ({len(tickers)} symbols) ===")
    print(f"enter_threshold={args.enter_threshold}, sizing_factor={args.sizing_factor}")
    print(f"mdd_cap={args.mdd_cap*100:.0f}%, cooldown={args.cooldown_days}일")
    print()

    print("[Universe regime]")
    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)
    n_bull = (regimes == "bull").sum()
    n_bear = (regimes == "bear").sum()
    print(f"  bull {n_bull}일 / bear {n_bear}일")
    print()

    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)

    # 1. 외국인만 vanilla
    foreign_strat = ForeignFlowProportional(
        sizing_factor=args.sizing_factor,
        min_threshold=args.enter_threshold,
        max_position_per_symbol=10_000,
    )
    res_foreign = run_portfolio_backtest(
        cfg, strategy=foreign_strat, initial_cash=args.initial_cash,
        signal_keys=("flow_ratio",),
    )

    # 2. 외국인만 + MDD cap
    foreign_mdd_strat = ForeignFlowProportional(
        sizing_factor=args.sizing_factor,
        min_threshold=args.enter_threshold,
        max_position_per_symbol=10_000,
    )
    res_foreign_mdd = run_portfolio_backtest(
        cfg, strategy=foreign_mdd_strat, initial_cash=args.initial_cash,
        mdd_cap=args.mdd_cap, cooldown_days=args.cooldown_days,
        signal_keys=("flow_ratio",),
    )

    # 3. 외국인+기관 동방향
    fi_strat = ForeignInstFlowFollow(
        enter_threshold=args.enter_threshold,
        sizing_factor=args.sizing_factor,
        min_threshold=0.01,
        max_position_per_symbol=10_000,
    )
    res_fi = run_portfolio_backtest(
        cfg, strategy=fi_strat, initial_cash=args.initial_cash,
        signal_keys=("flow_ratio", "inst_ratio"),
    )

    # 4. 외국인+기관 + MDD cap
    fi_mdd_strat = ForeignInstFlowFollow(
        enter_threshold=args.enter_threshold,
        sizing_factor=args.sizing_factor,
        min_threshold=0.01,
        max_position_per_symbol=10_000,
    )
    res_fi_mdd = run_portfolio_backtest(
        cfg, strategy=fi_mdd_strat, initial_cash=args.initial_cash,
        mdd_cap=args.mdd_cap, cooldown_days=args.cooldown_days,
        signal_keys=("flow_ratio", "inst_ratio"),
    )

    print("=" * 110)
    print("=== 비교 결과 ===")
    print("=" * 110)
    _print_header()
    _print_row("A. Buy & Hold", bh, regimes)
    _print_row("B. 외국인만 vanilla", res_foreign, regimes)
    _print_row("C. 외국인만 + MDD cap 25%", res_foreign_mdd, regimes)
    _print_row("D. 외국인+기관 동방향", res_fi, regimes)
    _print_row("E. 외국인+기관 + MDD cap 25%", res_fi_mdd, regimes)
    print()

    # save summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "tickers": tickers,
            "enter_threshold": args.enter_threshold,
            "sizing_factor": args.sizing_factor,
            "mdd_cap": args.mdd_cap,
        },
        "regime": {"bull": int(n_bull), "bear": int(n_bear)},
        "results": {
            "buyhold": {"final": bh.overall["final_value"], "return": bh.overall["return"], "mdd": bh.overall["max_drawdown"]},
            "foreign_only": {"final": res_foreign.overall["final_value"], "return": res_foreign.overall["return"], "mdd": res_foreign.overall["max_drawdown"]},
            "foreign_mdd": {"final": res_foreign_mdd.overall["final_value"], "return": res_foreign_mdd.overall["return"], "mdd": res_foreign_mdd.overall["max_drawdown"]},
            "foreign_inst": {"final": res_fi.overall["final_value"], "return": res_fi.overall["return"], "mdd": res_fi.overall["max_drawdown"]},
            "foreign_inst_mdd": {"final": res_fi_mdd.overall["final_value"], "return": res_fi_mdd.overall["return"], "mdd": res_fi_mdd.overall["max_drawdown"]},
        },
    }
    summary_path = out_dir / f"compare_foreign_inst_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
