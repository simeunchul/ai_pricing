"""Sell rule 가설 비교 백테스트.

가설 (사용자 2026-05-17, 5/12 MDD 사건 이후):
  양전(profit) 상태: peak_close 대비 -10% trailing stop
  음전(loss) 상태:   외인 OR 기관 한 쪽이라도 < -5% (weak signal)

비교 시나리오:
  V0  baseline                — 기존 dual AND sell + MDD cap (현 운영 룰)
  V1  + trailing 10% (양전만)
  V2  + weak OR (음전만)
  V3  V1 + V2 (사용자 가설 합본)
  V3 변형: trailing 5% / 7% / 15% sensitivity

매수 룰은 모든 시나리오에서 동일 (외인 AND 기관 둘 다 +5%).
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
    run_buyhold_portfolio,
    run_dual_dynamic_backtest,
    run_dual_dynamic_backtest_v2,
    universe_index, detect_regimes, analyze_regime_performance,
)


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 40 종 universe (기존 backtest_dynamic_universe.py 와 동일)
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


def _row(label: str, overall: dict, regimes_perf: dict | None = None) -> dict:
    calmar = _calmar(overall["return"], overall["max_drawdown"], overall["n_days"])
    r = {
        "label": label,
        "final": overall["final_value"],
        "return": overall["return"],
        "mdd": overall["max_drawdown"],
        "calmar": calmar,
        "n_days": overall["n_days"],
    }
    if regimes_perf:
        r["bull_cum"] = regimes_perf.get("bull", {}).get("cumulative_return", 0)
        r["bear_cum"] = regimes_perf.get("bear", {}).get("cumulative_return", 0)
    return r


def _print_header():
    print("=" * 120)
    print(f"  {'scenario':<40} {'final':>14} {'ret':>9} {'MDD':>8} {'Calmar':>8} {'bull':>9} {'bear':>9} {'sells':>6}")
    print("=" * 120)


def _print_row(r: dict, n_sells: int | None = None):
    bull = r.get("bull_cum", 0)
    bear = r.get("bear_cum", 0)
    sells = f"{n_sells}" if n_sells is not None else "—"
    print(f"  {r['label']:<40} {r['final']:>14,.0f} "
          f"{r['return']*100:>+8.2f}% {r['mdd']*100:>+7.2f}% "
          f"{r['calmar']:>+7.2f} {bull*100:>+8.2f}% {bear*100:>+8.2f}% {sells:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--enter-threshold", type=float, default=0.05)
    ap.add_argument("--max-concurrent", type=int, default=7)
    ap.add_argument("--mdd-cap", type=float, default=0.25,
                    help="0 이면 MDD cap 비활성")
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(
        symbols=tickers, cost_bps=args.cost_bps, max_pages=60,
    )

    print(f"=== Sell Rule Hypothesis Sweep ({len(tickers)} 종) ===")
    print(f"  enter_threshold={args.enter_threshold}, max_concurrent={args.max_concurrent}, "
          f"mdd_cap={args.mdd_cap}, initial={args.initial_cash:,.0f}")
    print()

    mdd_cap = args.mdd_cap if args.mdd_cap > 0 else None

    # universe regime
    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)
    print(f"[regime] bull {(regimes=='bull').sum()}일 / bear {(regimes=='bear').sum()}일")
    print()

    # Buy & Hold
    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)
    bh_perf = analyze_regime_performance(bh.history, regimes)
    bh_row = _row("0. Buy & Hold (40종 균등)", bh.overall, bh_perf)

    # 시나리오
    scenarios = [
        # V0 ~ V3: 사용자 가설 (이전 결과 재현)
        ("V0  baseline (dual AND + MDD)", dict()),
        ("V3  V1+V2 사용자 가설(trail 10+weak)", dict(trailing_stop_pct=0.10, weak_sell_when_loss=True)),

        # === Phase 2: 내 가설 — over-trade 안 하면서 black swan 보호 ===
        # A. Absolute stop loss (진입가 -X%) — 양/음전 무관, 단순/보수적
        ("A1  stop_loss -10%",            dict(stop_loss_pct=0.10)),
        ("A2  stop_loss -15%",            dict(stop_loss_pct=0.15)),
        ("A3  stop_loss -20%",            dict(stop_loss_pct=0.20)),

        # B. Trailing with N-day lock — whipsaw 보호 + 익절 보호
        ("B1  trailing 10% + lock 5일",   dict(trailing_stop_pct=0.10, trailing_lock_days=5)),
        ("B2  trailing 15% + lock 5일",   dict(trailing_stop_pct=0.15, trailing_lock_days=5)),
        ("B3  trailing 10% + lock 10일",  dict(trailing_stop_pct=0.10, trailing_lock_days=10)),
        ("B4  trailing 15% + lock 10일",  dict(trailing_stop_pct=0.15, trailing_lock_days=10)),

        # C. Time-based exit — dual rule exit signal 약점 보완
        ("C1  time exit 10일",            dict(time_exit_days=10)),
        ("C2  time exit 20일",            dict(time_exit_days=20)),
        ("C3  time exit 30일",            dict(time_exit_days=30)),

        # D. Best 후보 조합 (A2 + B2, A2 + C2 같은 결합)
        ("D1  stop 15% + trailing 15% lock 5일", dict(stop_loss_pct=0.15, trailing_stop_pct=0.15, trailing_lock_days=5)),
        ("D2  stop 15% + time 20",        dict(stop_loss_pct=0.15, time_exit_days=20)),

        # E. Asymmetric threshold (매수 +5% 고정, 매도만 변경)
        ("E1  sell threshold -3% (민감)", dict(sell_threshold_override=0.03)),
        ("E2  sell threshold -7%",        dict(sell_threshold_override=0.07)),
        ("E3  sell threshold -10% (둔감)", dict(sell_threshold_override=0.10)),
    ]

    results = []
    sell_reason_summary = []
    _print_header()
    _print_row(bh_row, n_sells=None)

    for label, kw in scenarios:
        res = run_dual_dynamic_backtest_v2(
            cfg, initial_cash=args.initial_cash,
            enter_threshold=args.enter_threshold,
            cost_bps=args.cost_bps,
            max_concurrent=args.max_concurrent,
            mdd_cap=mdd_cap, cooldown_days=0,
            **kw,
        )
        hist = res.history.copy()
        perf = analyze_regime_performance(hist, regimes)
        r = _row(label, res.overall, perf)
        results.append(r)

        sells = res.overall.get("sell_reasons", [])
        n_sells = len(sells)
        _print_row(r, n_sells=n_sells)

        if sells:
            from collections import Counter
            cnt = Counter(s["reason"] for s in sells)
            sell_reason_summary.append({
                "scenario": label,
                "n_total": n_sells,
                "by_reason": dict(cnt),
                "mean_pnl_by_reason": {
                    k: float(pd.Series([s["pnl"] for s in sells if s["reason"] == k]).mean())
                    for k in cnt
                },
            })

    print()
    print("[Sell reason breakdown — 시나리오별 매도 사유별 건수/평균 PnL]")
    for srs in sell_reason_summary:
        print(f"  {srs['scenario']}")
        print(f"    total {srs['n_total']} | "
              + " | ".join(f"{k}: {v}건 (avg PnL {srs['mean_pnl_by_reason'][k]*100:+.2f}%)"
                           for k, v in srs["by_reason"].items()))

    # save summary json
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "tickers": tickers, "n_tickers": len(tickers),
            "enter_threshold": args.enter_threshold,
            "max_concurrent": args.max_concurrent,
            "mdd_cap": mdd_cap,
            "cost_bps": args.cost_bps,
        },
        "buyhold": bh_row,
        "results": results,
        "sell_reason_summary": sell_reason_summary,
        "regime": {
            "bull_days": int((regimes == 'bull').sum()),
            "bear_days": int((regimes == 'bear').sum()),
        },
    }
    out_path = out_dir / f"sell_rule_hypothesis_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print()
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
