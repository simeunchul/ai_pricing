"""모멘텀 lookback 의 per-period + walk-forward 검증.

질문 (사용자 2026-06-09): 60일 lookback sweep 은 4년 in-sample 단일 백테스트라
과최적화 의심 + 최근 regime 이 과거 4년과 형태가 달라 신뢰 어려움.

(A) per-period: 데이터를 N 구간으로 쪼개 각 구간에서 lookback 별 성과 → 구간마다
    선호 lookback 이 다른지(=비정상성), 특히 '최근 구간'이 뭘 선호하는지 직접 확인.
(B) walk-forward: train 구간에서 Calmar 최고 lookback 선택 → 다음 test 구간(OOS)에
    적용, rolling. '과거에서 고른 값이 미래에 통하는가'를 검증. 고정 60일과 비교.

모든 백테스트는 라이브 운영룰(매도/replacement/MDD/10일 매수가드) 공통, momentum
lookback 만 변형.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig, run_dual_dynamic_backtest_v2,
    universe_index, detect_regimes,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_TICKERS = [
    "005930", "000660", "035420", "035720", "005380", "051910", "005490",
    "207940", "105560", "055550", "066570", "068270", "006400", "028260",
    "003670", "000270", "012330", "010130", "011170", "086790", "024110",
    "003490", "042660", "042700", "034020", "009830", "011200", "011780",
    "086280", "047810", "015760", "030200", "003550", "004020", "047040",
    "138930", "005935", "326030", "395400", "000720",
]

LIVE_BASE = dict(
    sell_threshold_override=0.07, replacement_pnl_max=0.02, replacement_min_hold=7,
    short_mom_lookback=10, short_mom_threshold=-0.05, sell_rule="dual", momentum_min=0.0,
    buy_short_mom_lookback=10, buy_short_mom_threshold=-0.05,
)
LOOKBACKS = [20, 30, 40, 60]


def _calmar(r, m, n):
    return float("nan") if (m == 0 or n <= 0) else ((1 + r) ** (252 / n) - 1) / abs(m)


def _run(cfg, lb, start, end):
    r = run_dual_dynamic_backtest_v2(
        cfg, initial_cash=1e7, enter_threshold=0.05, cost_bps=25,
        max_concurrent=7, mdd_cap=0.25, cooldown_days=0,
        momentum_lookback=lb, start_date=start, end_date=end, **LIVE_BASE,
    )
    o = r.overall
    if not o:
        return None
    return {"ret": o["return"], "mdd": o["max_drawdown"],
            "calmar": _calmar(o["return"], o["max_drawdown"], o["n_days"]), "n": o["n_days"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=6, help="per-period 구간 수")
    args = ap.parse_args()

    cfg = ForeignFlowBacktestConfig(symbols=DEFAULT_TICKERS, cost_bps=25, max_pages=60)
    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, 0.05)
    dates = uidx.index

    # 워밍업(60일) 이후부터 구간 분할
    span = dates[60:]
    bounds = [span[int(i * len(span) / args.slices)] for i in range(args.slices)] + [span[-1]]

    print("=" * 110)
    print(f"  (A) PER-PERIOD — 구간별 lookback Calmar (라이브룰 공통, 10일 매수가드 포함)")
    print("=" * 110)
    header = f"  {'구간':<26}{'bear%':>6}"
    for lb in LOOKBACKS:
        header += f"{str(lb)+'일':>9}"
    header += f"{'best':>7}"
    print(header)
    print("-" * 110)

    period_best = []
    for i in range(args.slices):
        s, e = bounds[i], bounds[i + 1]
        seg_reg = regimes[(regimes.index >= s) & (regimes.index <= e)]
        bear_pct = (seg_reg == "bear").mean() * 100 if len(seg_reg) else 0
        row = f"  {str(s.date())+'~'+str(e.date()):<26}{bear_pct:>5.0f}%"
        cals = {}
        for lb in LOOKBACKS:
            m = _run(cfg, lb, str(s.date()), str(e.date()))
            cals[lb] = m["calmar"] if m else float("nan")
            row += f"{cals[lb]:>+9.2f}"
        best_lb = max(cals, key=lambda k: (cals[k] if cals[k] == cals[k] else -9))
        period_best.append(best_lb)
        row += f"{str(best_lb)+'일':>7}"
        print(row)

    print("-" * 110)
    from collections import Counter
    print(f"  구간별 best lookback 분포: {dict(Counter(period_best))}")
    print(f"  → 최근 구간({str(bounds[-2].date())}~{str(bounds[-1].date())}) best = {period_best[-1]}일")

    # (B) Walk-forward: train=앞 2구간, test=다음 1구간, rolling
    print()
    print("=" * 110)
    print(f"  (B) WALK-FORWARD — train 2구간서 best 선택 → 다음 구간 OOS 적용 (고정 60일과 비교)")
    print("=" * 110)
    print(f"  {'test 구간':<26}{'선택lb(train기준)':>16}{'WF OOS Calmar':>15}{'고정60 OOS':>12}")
    print("-" * 110)
    wf_cals, fixed_cals = [], []
    for i in range(2, args.slices):
        tr_s, tr_e = bounds[i - 2], bounds[i]      # train: 2구간
        te_s, te_e = bounds[i], bounds[i + 1]       # test: 다음 1구간
        tr = {lb: _run(cfg, lb, str(tr_s.date()), str(tr_e.date())) for lb in LOOKBACKS}
        pick = max(LOOKBACKS, key=lambda lb: (tr[lb]["calmar"] if tr[lb] and tr[lb]["calmar"] == tr[lb]["calmar"] else -9))
        wf = _run(cfg, pick, str(te_s.date()), str(te_e.date()))
        fx = _run(cfg, 60, str(te_s.date()), str(te_e.date()))
        wf_cals.append(wf["calmar"]); fixed_cals.append(fx["calmar"])
        print(f"  {str(te_s.date())+'~'+str(te_e.date()):<26}{str(pick)+'일':>16}"
              f"{wf['calmar']:>+15.2f}{fx['calmar']:>+12.2f}")
    print("-" * 110)
    wf_avg = pd.Series(wf_cals).mean(); fx_avg = pd.Series(fixed_cals).mean()
    print(f"  OOS 평균 Calmar — WF적응형 {wf_avg:+.2f}  vs  고정60일 {fx_avg:+.2f}  "
          f"→ {'WF 우세' if wf_avg > fx_avg else '고정60 우세 (적응 무의미)'}")


if __name__ == "__main__":
    main()
