"""매수 모멘텀 게이트 변형 비교 백테스트.

질문 (사용자 2026-06-08): "듀얼신호(외인+기관)가 이미 들어온 후보는 매수하고,
추세는 보조로 쓰면 어떤가?" — 현재는 60일 추세 > 0 을 하드 게이트로 써서 추세
음전 종목(예: 카카오뱅크 -2.95%)을 완전히 거부함.

baseline 은 현 라이브 운영 룰과 동일하게 맞춤 (매도 룰·replacement·MDD cap 등
모든 시나리오 공통). 오직 "매수 모멘텀 게이트"만 변형해 Calmar·MDD·수익률·현금
활용률을 비교한다.

변형:
  baseline  하드 게이트 (momentum_min=0.0)          ← 현 운영
  ②a/b/c    Floor (-0.03 / -0.05 / -0.10)           ← 약한 음전만 허용 (가드레일)
  ③         Dual-only (게이트 OFF)                   ← 추세 완전 무시
  ①a/b      Soft sizing (음전 종목 0.5 / 0.3 배 축소 진입)

주의: 매도 룰 중 #5 코스피-underperform 청산은 백테스트 엔진에 없음. 단 모든
시나리오에 공통으로 부재하므로 변형 간 상대 비교는 유효하다.
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
    ForeignFlowBacktestConfig,
    run_buyhold_portfolio,
    run_dual_dynamic_backtest_v2,
    universe_index, detect_regimes, analyze_regime_performance,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# backtest_dual_sell_rule.py 와 동일한 40종 universe
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

# 현 라이브 운영 룰 (run_dual_paper_trading.py 상수와 일치) — 모든 시나리오 공통.
LIVE_BASE = dict(
    sell_threshold_override=0.07,   # SELL_THRESHOLD (비대칭 매도)
    replacement_pnl_max=0.02,       # REPLACEMENT_PNL_MAX
    replacement_min_hold=7,         # REPLACEMENT_MIN_HOLD
    short_mom_lookback=10,          # A3 매도 SHORT_MOM_LOOKBACK
    short_mom_threshold=-0.05,      # A3 매도 SHORT_MOM_THRESHOLD
    sell_rule="dual",
)

# 매수 모멘텀 게이트 변형만 다름.
VARIANTS = [
    ("baseline 하드게이트 (min=0.0, 현 운영)", dict(momentum_lookback=60, momentum_min=0.0)),
    ("②a Floor min=-0.03",                     dict(momentum_lookback=60, momentum_min=-0.03)),
    ("②b Floor min=-0.05",                     dict(momentum_lookback=60, momentum_min=-0.05)),
    ("②c Floor min=-0.10",                     dict(momentum_lookback=60, momentum_min=-0.10)),
    ("③ Dual-only (게이트 OFF)",               dict(momentum_lookback=None)),
    ("①a Soft sizing 0.5 (음전 절반)",         dict(momentum_lookback=60, momentum_min=0.0, momentum_soft_factor=0.5)),
    ("①b Soft sizing 0.3 (음전 30%)",          dict(momentum_lookback=60, momentum_min=0.0, momentum_soft_factor=0.3)),
]


def _calmar(total_return: float, mdd: float, n_days: int) -> float:
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized / abs(mdd)


def _utilization(history: pd.DataFrame) -> tuple[float, float]:
    """평균 투자비중(1 - cash/value) 과 평균 보유종목수 — '돈 굴린 정도'."""
    tot = history[history["symbol"] == "_TOTAL_"].copy()
    if tot.empty:
        return float("nan"), float("nan")
    tot = tot[tot["value"] > 0]
    invested = (1.0 - tot["cash"] / tot["value"]).mean()
    avg_pos = tot["position"].mean()
    return float(invested), float(avg_pos)


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

    print("=" * 132)
    print(f"  매수 모멘텀 게이트 변형 비교 ({len(tickers)}종)")
    print(f"  공통: enter={args.enter_threshold}, max_conc={args.max_concurrent}, "
          f"mdd_cap={args.mdd_cap}, cost={args.cost_bps}bps, A3매도+replacement 동일")
    print(f"  데이터 구간 길이: {len(uidx)}일 / bull {(regimes=='bull').sum()}일 · bear {(regimes=='bear').sum()}일")
    print("=" * 132)
    print(f"  {'변형':<34} {'final':>13} {'ret':>9} {'MDD':>8} {'Calmar':>8} "
          f"{'bull':>8} {'bear':>8} {'투자비중':>9} {'평균보유':>8} {'sells':>6}")
    print("-" * 132)

    bh = run_buyhold_portfolio(cfg, initial_cash=args.initial_cash)
    bh_perf = analyze_regime_performance(bh.history, regimes)
    print(f"  {'0. Buy&Hold (균등)':<34} {bh.overall['final_value']:>13,.0f} "
          f"{bh.overall['return']*100:>+8.2f}% {bh.overall['max_drawdown']*100:>+7.2f}% "
          f"{_calmar(bh.overall['return'], bh.overall['max_drawdown'], bh.overall['n_days']):>+7.2f} "
          f"{bh_perf.get('bull',{}).get('cumulative_return',0)*100:>+7.2f}% "
          f"{bh_perf.get('bear',{}).get('cumulative_return',0)*100:>+7.2f}% "
          f"{'—':>9} {'—':>8} {'—':>6}")

    results = []
    for label, kw in VARIANTS:
        res = run_dual_dynamic_backtest_v2(
            cfg, initial_cash=args.initial_cash,
            enter_threshold=args.enter_threshold,
            cost_bps=args.cost_bps,
            max_concurrent=args.max_concurrent,
            mdd_cap=mdd_cap, cooldown_days=0,
            **LIVE_BASE, **kw,
        )
        perf = analyze_regime_performance(res.history, regimes)
        cal = _calmar(res.overall["return"], res.overall["max_drawdown"], res.overall["n_days"])
        invested, avg_pos = _utilization(res.history)
        sells = res.overall.get("sell_reasons", [])
        bull = perf.get("bull", {}).get("cumulative_return", 0)
        bear = perf.get("bear", {}).get("cumulative_return", 0)
        print(f"  {label:<34} {res.overall['final_value']:>13,.0f} "
              f"{res.overall['return']*100:>+8.2f}% {res.overall['max_drawdown']*100:>+7.2f}% "
              f"{cal:>+7.2f} {bull*100:>+7.2f}% {bear*100:>+7.2f}% "
              f"{invested*100:>+8.1f}% {avg_pos:>7.1f} {len(sells):>6}")
        results.append({
            "label": label, "params": kw,
            "final": res.overall["final_value"], "return": res.overall["return"],
            "mdd": res.overall["max_drawdown"], "calmar": cal,
            "bull": bull, "bear": bear,
            "invested_ratio": invested, "avg_positions": avg_pos,
            "n_sells": len(sells),
            "sell_by_reason": dict(Counter(s["reason"] for s in sells)),
        })

    print("=" * 132)
    print("  투자비중 = 평균 (1 - 현금/총자산), 높을수록 돈을 더 굴림 / 평균보유 = 평균 동시 보유 종목수")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"momentum_gate_compare_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"tickers": tickers, "n_tickers": len(tickers),
                       "enter_threshold": args.enter_threshold,
                       "max_concurrent": args.max_concurrent, "mdd_cap": mdd_cap,
                       "cost_bps": args.cost_bps, "live_base": LIVE_BASE,
                       "window_days": len(uidx),
                       "bull_days": int((regimes == 'bull').sum()),
                       "bear_days": int((regimes == 'bear').sum())},
            "results": results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
