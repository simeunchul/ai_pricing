"""ForeignFlowProportional 종합 비교 — Universe 확장 + 모든 risk control + Regime split.

비교 대상 5가지:
  A. Buy & Hold (baseline)
  B. 외국인 추종 (vanilla)
  C. + Volatility-based sizing
  D. + Portfolio MDD Cap 20%
  E. + Vol sizing + MDD cap (combined)

각 변형에 대해:
  - Final value, return, MDD, Calmar
  - Universe 인덱스 기반 regime (bull/bear) 분리해서 각 구간 cumulative return / Sharpe

Usage:
  python scripts/backtest_foreign_flow_full.py
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


# 강세 + 약세 + 다양한 산업군 mix (15종)
DEFAULT_TICKERS = [
    "005930",  # 삼성전자 (반도체 강세)
    "000660",  # SK하이닉스 (반도체 강세)
    "035420",  # NAVER (IT 약세)
    "035720",  # 카카오 (IT 약세)
    "005380",  # 현대차 (자동차)
    "051910",  # LG화학 (화학)
    "005490",  # POSCO홀딩스 (철강)
    "207940",  # 삼성바이오로직스 (바이오)
    "105560",  # KB금융 (금융)
    "055550",  # 신한지주 (금융)
    "066570",  # LG전자 (전자)
    "068270",  # 셀트리온 (바이오)
    "006400",  # 삼성SDI (배터리 약세)
    "028260",  # 삼성물산 (지주)
    "003670",  # 포스코퓨처엠 (배터리 약세)
]


def _calmar(total_return: float, mdd: float, n_days: int) -> float:
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    if years <= 0:
        return float("nan")
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized / abs(mdd)


def _print_metrics(label: str, res, regimes):
    o = res.overall
    calmar = _calmar(o["return"], o["max_drawdown"], o["n_days"])
    print(f"\n--- {label} ---")
    print(f"  final           : {o['final_value']:>14,.0f} 원")
    print(f"  return          : {o['return']*100:>+13.2f}%")
    print(f"  MDD             : {o['max_drawdown']*100:>+13.2f}%")
    print(f"  Calmar          : {calmar:>+13.2f}")

    perf = analyze_regime_performance(res.history, regimes)
    bull = perf.get("bull", {})
    bear = perf.get("bear", {})
    if bull.get("n_days", 0) > 0:
        print(f"  bull ({bull['n_days']:>3}d) : "
              f"cum={bull['cumulative_return']*100:>+8.2f}%  "
              f"Sharpe(ann)={bull.get('sharpe_annualized', 0):>+5.2f}")
    if bear.get("n_days", 0) > 0:
        print(f"  bear ({bear['n_days']:>3}d) : "
              f"cum={bear['cumulative_return']*100:>+8.2f}%  "
              f"Sharpe(ann)={bear.get('sharpe_annualized', 0):>+5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--sizing-factor", type=float, default=200.0)
    ap.add_argument("--min-threshold", type=float, default=0.05)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--vol-target", type=float, default=0.02,
                    help="목표 일별 std (default 2%)")
    ap.add_argument("--mdd-cap", type=float, default=0.30,
                    help="portfolio MDD cap (default 30%)")
    ap.add_argument("--cooldown-days", type=int, default=10,
                    help="MDD cap 발동 후 진입 차단 일수 (default 10)")
    ap.add_argument("--bear-threshold", type=float, default=0.05,
                    help="bear 구간 정의 drawdown 임계 (default 5%)")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(
        symbols=tickers, cost_bps=args.cost_bps, max_pages=args.max_pages,
    )

    print("=== ForeignFlow Full Comparison — 15 Symbols + Risk Controls + Regime ===")
    print(f"tickers      : {len(tickers)} 종")
    print(f"sf={args.sizing_factor}  th={args.min_threshold}  "
          f"vol_target={args.vol_target}  mdd_cap={args.mdd_cap}")
    print(f"initial cash : {args.initial_cash:,.0f} 원 "
          f"(per-symbol {args.initial_cash/len(tickers):,.0f}원)")
    print(f"bear def     : universe drawdown ≥ {args.bear_threshold*100:.0f}%")
    print()

    print("[1/6] Universe index 계산 + bear/bull regime 분리...")
    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=args.bear_threshold)
    n_bull = (regimes == "bull").sum()
    n_bear = (regimes == "bear").sum()
    print(f"  total {len(regimes)} 거래일: bull {n_bull} / bear {n_bear}")

    # Universe 인덱스의 누적 변화
    if not uidx.empty:
        idx_start = float(uidx.iloc[0])
        idx_end = float(uidx.iloc[-1])
        idx_max = float(uidx.cummax().iloc[-1])
        idx_mdd = float(((uidx - uidx.cummax()) / uidx.cummax()).min())
        print(f"  universe index : start=1.00, end={idx_end:.2f} "
              f"({(idx_end-1)*100:+.1f}%), MDD={idx_mdd*100:+.1f}%")

    def make_strat():
        return ForeignFlowProportional(
            sizing_factor=args.sizing_factor,
            min_threshold=args.min_threshold,
            max_position_per_symbol=10_000,
        )

    print("\n[2/6] A. Buy & Hold baseline...")
    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)
    _print_metrics("A. Buy & Hold (baseline)", bh, regimes)

    print("\n[3/6] B. 외국인 추종 vanilla...")
    res_b = run_portfolio_backtest(cfg, strategy=make_strat(),
                                     initial_cash=args.initial_cash)
    _print_metrics("B. 외국인 추종 vanilla", res_b, regimes)

    print("\n[4/6] C. + Volatility-based sizing...")
    res_c = run_portfolio_backtest(cfg, strategy=make_strat(),
                                     initial_cash=args.initial_cash,
                                     vol_target_daily=args.vol_target,
                                     vol_lookback=20)
    _print_metrics(f"C. + Vol sizing (target={args.vol_target})", res_c, regimes)

    print("\n[5/6] D. + Portfolio MDD Cap...")
    res_d = run_portfolio_backtest(cfg, strategy=make_strat(),
                                     initial_cash=args.initial_cash,
                                     mdd_cap=args.mdd_cap,
                                     cooldown_days=args.cooldown_days)
    _print_metrics(f"D. + MDD Cap ({args.mdd_cap*100:.0f}%)", res_d, regimes)

    print("\n[6/6] E. + Vol sizing + MDD Cap (combined)...")
    res_e = run_portfolio_backtest(cfg, strategy=make_strat(),
                                     initial_cash=args.initial_cash,
                                     vol_target_daily=args.vol_target,
                                     vol_lookback=20,
                                     mdd_cap=args.mdd_cap,
                                     cooldown_days=args.cooldown_days)
    _print_metrics("E. + Vol sizing + MDD Cap (combined)", res_e, regimes)

    # === Summary ===
    print("\n" + "=" * 80)
    print("=== SUMMARY ===")
    print("=" * 80)
    print(f"  {'전략':<35} {'final':>14} {'ret':>9} {'MDD':>8} {'Calmar':>8}")
    for label, res in [
        ("A. Buy & Hold (baseline)", bh),
        ("B. 외국인 추종 vanilla", res_b),
        ("C. + Volatility sizing", res_c),
        ("D. + MDD cap", res_d),
        ("E. + Vol + MDD (combined)", res_e),
    ]:
        o = res.overall
        calmar = _calmar(o["return"], o["max_drawdown"], o["n_days"])
        print(f"  {label:<35} {o['final_value']:>14,.0f} "
              f"{o['return']*100:>+8.2f}% {o['max_drawdown']*100:>+7.2f}% "
              f"{calmar:>+7.2f}")

    print()
    print("=== REGIME COMPARISON (bear 구간 cumulative return) ===")
    print(f"  {'전략':<35} {'bull cum':>10} {'bear cum':>10} {'bear/bull':>10}")
    for label, res in [
        ("A. Buy & Hold", bh),
        ("B. vanilla", res_b),
        ("C. + Vol", res_c),
        ("D. + MDD", res_d),
        ("E. + Vol + MDD", res_e),
    ]:
        perf = analyze_regime_performance(res.history, regimes)
        bull_cum = perf.get("bull", {}).get("cumulative_return", 0)
        bear_cum = perf.get("bear", {}).get("cumulative_return", 0)
        ratio = bear_cum / bull_cum if bull_cum != 0 else float("nan")
        print(f"  {label:<35} {bull_cum*100:>+9.2f}% {bear_cum*100:>+9.2f}% "
              f"{ratio:>+9.2f}")

    # Save histories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "tickers": tickers,
            "sizing_factor": args.sizing_factor,
            "min_threshold": args.min_threshold,
            "vol_target": args.vol_target,
            "mdd_cap": args.mdd_cap,
            "bear_threshold": args.bear_threshold,
        },
        "regime": {"bull_days": int(n_bull), "bear_days": int(n_bear)},
        "results": {
            label: {
                "final_value": res.overall["final_value"],
                "return": res.overall["return"],
                "max_drawdown": res.overall["max_drawdown"],
                "regime": analyze_regime_performance(res.history, regimes),
            }
            for label, res in [
                ("buyhold", bh), ("vanilla", res_b),
                ("vol_sizing", res_c), ("mdd_cap", res_d),
                ("combined", res_e),
            ]
        },
    }
    summary_path = out_dir / f"backtest_full_comparison_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] summary : {summary_path}")


if __name__ == "__main__":
    main()
