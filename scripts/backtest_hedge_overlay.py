"""인버스 헤지 오버레이 백테스트 (순수헤지 / 전술 × −1x / −2x).

질문 (사용자 2026-06-08): 듀얼 롱 전략은 롱온리라 하락장에 기껏 본전(+2%).
하락장에서도 +수익을 내려면 인버스(반대 방향) 포지션을 얹자.

모델: 기존 듀얼 전략의 일별 자산곡선 위에 인버스 슬리브를 오버레이.
  - 하락 regime(유니버스 지수 고점 대비 낙폭 ≥ 5%)일 때만 인버스 비중 w 적용.
  - look-ahead 방지: 헤지 결정은 전일(regime[t-1]) 기준.
  - 일별 포트 수익 = (1-w)·전략수익 + w·인버스수익,
    인버스수익 = -leverage × 지수일수익 - 보수(expense) ; regime 토글 시 거래비용.
  - 인버스 ETF 변동성 침식(decay)은 일별 복리로 자연 반영, -2x 에서 커짐.

변형 (net beta = bear 국면의 시장 순노출, 전략 beta≈1 가정):
  baseline      헤지 없음
  순수헤지 −1x   w=0.50, k=1 → net ≈ 0   (시장 중립, 본전 방어)
  순수헤지 −2x   w=0.33, k=2 → net ≈ 0
  전술 −1x       w=1.00, k=1 → net ≈ -1  (완전 인버스, 하락장 +수익 추구)
  전술 −2x       w=1.00, k=2 → net ≈ -2  (가장 공격적, decay 위험 큼)

주의: 인버스 ETF 실가격이 아니라 지수 -k배로 합성 모델링. 실제 ETF 추적오차·
괴리율은 미반영(보수적으로 expense+toggle 비용만). regime 판정은 backtest 가
전체 지수를 보고 causal drawdown 으로 계산(약한 look-ahead 가능) → 1일 lag 로 완화.
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
    run_dual_dynamic_backtest_v2,
    universe_index, detect_regimes,
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

LIVE_BASE = dict(
    sell_threshold_override=0.07, replacement_pnl_max=0.02, replacement_min_hold=7,
    short_mom_lookback=10, short_mom_threshold=-0.05, sell_rule="dual",
    momentum_lookback=60, momentum_min=0.0,
)

# (label, w_bear, leverage)
HEDGE_VARIANTS = [
    ("0. 헤지 없음 (현 운영)",        0.00, 1),
    ("순수헤지 −1x (net≈0)",         0.50, 1),
    ("순수헤지 −2x (net≈0)",         0.33, 2),
    ("전술 −1x (net≈-1, 하락수익)",   1.00, 1),
    ("전술 −2x (net≈-2, 공격적)",     1.00, 2),
]

EXPENSE_ANNUAL = 0.007       # 인버스 ETF 보수 ~0.7%/yr
TOGGLE_COST = 0.0025         # regime 토글 시 |Δw| 당 25bps 거래비용


def _bear_signal(uidx: pd.Series, mode: str) -> pd.Series:
    """헤지 트리거 — 'bear' 인 날에만 인버스 발동.

    drawdown : 고점 대비 -5% (느림 — 회복 랠리도 bear 로 잡아 숏 유지 → 나쁨)
    ma20/ma60: 지수 < N일 이동평균 (추세추종 — 반등 시 즉시 bull 복귀)
    """
    if mode == "drawdown":
        return detect_regimes(uidx, drawdown_threshold=0.05)
    n = int(mode.replace("ma", ""))
    ma = uidx.rolling(n).mean()
    return pd.Series(["bear" if (not pd.isna(m)) and v < m else "bull"
                      for v, m in zip(uidx, ma)], index=uidx.index)


def _calmar(total_return: float, mdd: float, n_days: int) -> float:
    if mdd == 0 or n_days <= 0:
        return float("nan")
    annualized = (1 + total_return) ** (252 / n_days) - 1
    return annualized / abs(mdd)


def _metrics(equity: pd.Series, regimes: pd.Series, initial: float) -> dict:
    ret = float(equity.iloc[-1] / initial - 1)
    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min())
    cal = _calmar(ret, mdd, len(equity))
    daily = equity.pct_change().dropna()
    daily.index = pd.to_datetime(daily.index).normalize()
    reg = regimes.copy(); reg.index = pd.to_datetime(reg.index).normalize()
    common = daily.index.intersection(reg.index)
    daily, reg = daily.loc[common], reg.loc[common]
    bull = float((1 + daily[reg == "bull"]).prod() - 1)
    bear = float((1 + daily[reg == "bear"]).prod() - 1)
    return {"final": float(equity.iloc[-1]), "return": ret, "mdd": mdd,
            "calmar": cal, "bull": bull, "bear": bear}


def _apply_hedge(strat_equity: pd.Series, index_ret: pd.Series,
                 regimes_lagged: pd.Series, w_bear: float, k: int,
                 initial: float) -> pd.Series:
    strat_ret = strat_equity.pct_change().fillna(0.0)
    expense_daily = EXPENSE_ANNUAL / 252.0
    eq = initial
    prev_w = 0.0
    out = {}
    for d in strat_equity.index:
        reg = regimes_lagged.get(d, "bull")
        w = w_bear if reg == "bear" else 0.0
        ir = index_ret.get(d, 0.0)
        if pd.isna(ir):
            ir = 0.0
        hedge_ret = -k * ir - expense_daily
        port_ret = (1 - w) * float(strat_ret[d]) + w * hedge_ret
        port_ret -= abs(w - prev_w) * TOGGLE_COST
        eq *= (1 + port_ret)
        out[d] = eq
        prev_w = w
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--enter-threshold", type=float, default=0.05)
    ap.add_argument("--max-concurrent", type=int, default=7)
    ap.add_argument("--mdd-cap", type=float, default=0.25)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--trigger", default="drawdown", choices=["drawdown", "ma20", "ma60"],
                    help="헤지 발동 트리거 (drawdown=고점-5%, ma20/ma60=이평선 하향)")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(symbols=tickers, cost_bps=args.cost_bps, max_pages=60)
    mdd_cap = args.mdd_cap if args.mdd_cap > 0 else None

    uidx = universe_index(cfg)
    uidx.index = pd.to_datetime(uidx.index).normalize()
    regimes = _bear_signal(uidx, args.trigger)
    regimes_lagged = regimes.shift(1).fillna("bull")   # look-ahead 방지
    index_ret = uidx.pct_change()

    n_bear = int((regimes == "bear").sum())
    n_bull = int((regimes == "bull").sum())

    # 전략 1회 실행 → 자산곡선
    res = run_dual_dynamic_backtest_v2(
        cfg, initial_cash=args.initial_cash, enter_threshold=args.enter_threshold,
        cost_bps=args.cost_bps, max_concurrent=args.max_concurrent,
        mdd_cap=mdd_cap, cooldown_days=0, **LIVE_BASE,
    )
    tot = res.history[res.history["symbol"] == "_TOTAL_"].copy()
    tot["date"] = pd.to_datetime(tot["date"]).dt.normalize()
    strat_equity = tot.set_index("date")["value"].sort_index()

    print("=" * 104)
    print(f"  인버스 헤지 오버레이 비교 ({len(tickers)}종)")
    print(f"  데이터 {len(uidx)}일 / bull {n_bull}일 · bear {n_bear}일 "
          f"({n_bear/max(1,len(uidx))*100:.0f}% 하락국면) | expense {EXPENSE_ANNUAL*100:.1f}%/yr, toggle {TOGGLE_COST*1e4:.0f}bps")
    print("=" * 104)
    print(f"  {'변형':<30} {'final':>14} {'ret':>9} {'MDD':>8} {'Calmar':>8} {'bull':>9} {'bear':>9}")
    print("-" * 104)

    results = []
    for label, w, k in HEDGE_VARIANTS:
        if w == 0.0:
            eq = strat_equity.copy()
        else:
            eq = _apply_hedge(strat_equity, index_ret, regimes_lagged, w, k, args.initial_cash)
        m = _metrics(eq, regimes, args.initial_cash)
        print(f"  {label:<30} {m['final']:>14,.0f} {m['return']*100:>+8.2f}% "
              f"{m['mdd']*100:>+7.2f}% {m['calmar']:>+7.2f} "
              f"{m['bull']*100:>+8.2f}% {m['bear']*100:>+8.2f}%")
        results.append({"label": label, "w_bear": w, "leverage": k, **m})

    print("=" * 104)
    print("  bear 양수 = 하락국면에서 +수익. 단 MDD·Calmar 동반 확인 필수 (헤지는 상승장 비용 발생)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hedge_overlay_compare_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"n_tickers": len(tickers), "window_days": len(uidx),
                              "bull_days": n_bull, "bear_days": n_bear,
                              "expense_annual": EXPENSE_ANNUAL, "toggle_cost": TOGGLE_COST},
                   "results": results}, f, ensure_ascii=False, indent=2, default=str)
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
