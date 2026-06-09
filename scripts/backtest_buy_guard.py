"""매수 단기추세 가드 + 재매수 쿨다운 백테스트 (churn 차단 검증).

문제 (사용자 2026-06-09, 028050 사례): 매수 게이트는 60일 추세만 보고, A3 매도는
10일 추세를 본다. 그래서 60일은 +지만 10일은 급락 중인 종목을 사자마자 A3가 되팔아
수수료만 날리는 churn(휩쏘)이 난다. A3 매도 룰은 유지하고(30위 밖 추세매도 필요),
매수 쪽만 손봐서 churn 을 막을 수 있는지 검증한다.

Fix 1: 매수 게이트에 10일 단기추세 가드 (10일 < 임계면 매수 보류)
Fix 2: 재매수 쿨다운 (판 종목 N일 재매수 금지)

baseline 은 현 라이브 운영 룰과 동일. 모든 변형에서 A3 매도·replacement·MDD cap 불변.
churn 지표 = 보유 2일 이내 매도 건수 + A3 매도 건수.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

import pandas as pd

from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig, run_dual_dynamic_backtest_v2,
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

# 현 라이브 운영 룰 (run_dual_paper_trading.py 와 일치) — 모든 변형 공통.
LIVE_BASE = dict(
    sell_threshold_override=0.07, replacement_pnl_max=0.02, replacement_min_hold=7,
    short_mom_lookback=10, short_mom_threshold=-0.05, sell_rule="dual",
    momentum_lookback=60, momentum_min=0.0,
)

VARIANTS = [
    ("baseline (현 운영)",                dict()),
    ("Fix1 매수가드 10일 ≥ -3%",          dict(buy_short_mom_lookback=10, buy_short_mom_threshold=-0.03)),
    ("Fix1 매수가드 10일 ≥ -5%",          dict(buy_short_mom_lookback=10, buy_short_mom_threshold=-0.05)),
    ("Fix1 매수가드 10일 ≥ -7%",          dict(buy_short_mom_lookback=10, buy_short_mom_threshold=-0.07)),
    ("Fix2 재매수쿨다운 2일",             dict(rebuy_cooldown_days=2)),
    ("Fix2 재매수쿨다운 3일",             dict(rebuy_cooldown_days=3)),
    ("Fix2 재매수쿨다운 5일",             dict(rebuy_cooldown_days=5)),
    ("Fix1+2 가드-5% + 쿨다운3일",        dict(buy_short_mom_lookback=10, buy_short_mom_threshold=-0.05, rebuy_cooldown_days=3)),
]


def _calmar(r: float, m: float, n: int) -> float:
    if m == 0 or n <= 0:
        return float("nan")
    return ((1 + r) ** (252 / n) - 1) / abs(m)


def _churn_stats(sells: list) -> dict:
    n = len(sells)
    a3 = [s for s in sells if s.get("reason") == "short_momentum"]
    churn = [s for s in sells if s.get("hold_days", 99) <= 2]   # 보유 2일 이내 = churn
    churn_pnl = pd.Series([s["pnl"] for s in churn]).mean() if churn else float("nan")
    return {"n_sells": n, "n_a3": len(a3), "n_churn": len(churn),
            "churn_mean_pnl": float(churn_pnl) if churn else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--enter-threshold", type=float, default=0.05)
    ap.add_argument("--max-concurrent", type=int, default=7)
    ap.add_argument("--mdd-cap", type=float, default=0.25)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(symbols=tickers, cost_bps=args.cost_bps, max_pages=60)
    mdd_cap = args.mdd_cap if args.mdd_cap > 0 else None
    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)

    print("=" * 124)
    print(f"  매수 가드 + 재매수 쿨다운 비교 ({len(tickers)}종, {len(uidx)}일) — A3 매도 등 방어룰 전부 공통")
    print("=" * 124)
    print(f"  {'변형':<30} {'ret':>9} {'MDD':>8} {'Calmar':>7} {'bear':>8} "
          f"{'총매도':>6} {'A3매도':>6} {'churn(≤2일)':>11} {'churn평균pnl':>11}")
    print("-" * 124)

    results = []
    for label, kw in VARIANTS:
        res = run_dual_dynamic_backtest_v2(
            cfg, initial_cash=args.initial_cash, enter_threshold=args.enter_threshold,
            cost_bps=args.cost_bps, max_concurrent=args.max_concurrent,
            mdd_cap=mdd_cap, cooldown_days=0, **LIVE_BASE, **kw,
        )
        o = res.overall
        perf = analyze_regime_performance(res.history, regimes)
        cs = _churn_stats(o.get("sell_reasons", []))
        cal = _calmar(o["return"], o["max_drawdown"], o["n_days"])
        bear = perf.get("bear", {}).get("cumulative_return", 0)
        cpnl = f"{cs['churn_mean_pnl']*100:+.2f}%" if cs["churn_mean_pnl"] is not None else "—"
        print(f"  {label:<30} {o['return']*100:>+8.1f}% {o['max_drawdown']*100:>+7.1f}% "
              f"{cal:>+6.2f} {bear*100:>+7.1f}% {cs['n_sells']:>6} {cs['n_a3']:>6} "
              f"{cs['n_churn']:>11} {cpnl:>11}")
        results.append({"label": label, "params": kw, "return": o["return"],
                        "mdd": o["max_drawdown"], "calmar": cal, "bear": bear, **cs})

    print("=" * 124)
    print("  churn(≤2일) = 산 지 2거래일 이내 되판 건수(=휩쏘). 낮을수록 좋음. 수익/Calmar 동반 확인.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_dir) / f"buy_guard_compare_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"n_tickers": len(tickers), "window_days": len(uidx),
                              "live_base": LIVE_BASE}, "results": results},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
