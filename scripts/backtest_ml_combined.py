"""ML score + Dual Confirmation 결합 백테스트.

ML 모델 (학습 완료) 의 일별 점수를 사용해 dual confirmation 통과 종목 중
점수 높은 종목만 진입 (또는 추가 필터).

비교:
  A. Dual only (현재 best) — mdd_cap 30%, conc 7
  B. Dual + ML filter (>= threshold)
  C. ML only (참고, dual 무시)
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
import numpy as np

from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig, _load_symbol_data, _common_date_range,
    run_buyhold_portfolio, PortfolioBacktestResult,
    universe_index, detect_regimes, analyze_regime_performance,
)
from autotrader.ml import build_features, load_model, predict_score, FEATURE_COLS

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_TICKERS = [
    "005930","000660","035420","035720","005380","051910","005490","207940",
    "105560","055550","066570","068270","006400","028260","003670",
    "000270","012330","010130","011170","086790","024110","003490","042660",
    "042700","034020","009830","011200","011780","086280","047810","015760",
    "030200","003550","004020","047040","138930","005935","326030","395400","000720",
]


def run_combined_backtest(
    cfg: ForeignFlowBacktestConfig,
    model,
    test_only: bool = True,
    train_split_date: str = "2024-09-01",
    initial_cash: float = 10_000_000.0,
    enter_threshold: float = 0.05,
    max_concurrent: int = 7,
    mdd_cap: float | None = 0.30,
    cooldown_days: int = 10,
    ml_threshold: float | None = None,    # None=비활성, dual only
    use_ml_only: bool = False,             # True 면 dual 무시하고 ML 단독
    cost_bps: float = 25.0,
) -> PortfolioBacktestResult:
    """동적 universe + dual + (optional) ML score 필터.

    test_only=True: train_split_date 이후만 백테스트 (out-of-sample)
    """
    cost_per_side = cost_bps / 10_000.0 / 2.0

    sym_data: dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        df = _load_symbol_data(sym, cfg)
        if not df.empty:
            sym_data[sym] = df
    if not sym_data:
        return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})

    rng = _common_date_range(sym_data)
    common_start, common_end = rng

    # Features 미리 빌드 (모든 종목, 모든 날짜)
    from autotrader.market.chart_indicators import add_chart_features
    from autotrader.ml.dual_signal_classifier import _add_flow_ml_features

    enriched = {}
    for sym, df in sym_data.items():
        with_chart = add_chart_features(df)
        enriched[sym] = _add_flow_ml_features(with_chart)

    # ML 점수 계산 — 모든 종목 × 모든 날짜
    score_lookup: dict[tuple[str, pd.Timestamp], float] = {}
    if ml_threshold is not None or use_ml_only:
        for sym, df in enriched.items():
            valid = df.dropna(subset=FEATURE_COLS)
            if valid.empty:
                continue
            scores = predict_score(model, valid)
            for d, s in zip(valid.index, scores):
                score_lookup[(sym, d)] = float(s)

    all_dates = sorted([
        d for d in set().union(*[set(df.index) for df in sym_data.values()])
        if common_start <= d <= common_end
    ])
    # test_only: train 기간 skip
    if test_only:
        cutoff = pd.Timestamp(train_split_date)
        all_dates = [d for d in all_dates if d >= cutoff]
        if len(all_dates) < 2:
            return PortfolioBacktestResult(pd.DataFrame(), pd.DataFrame(), {})

    positions = {}
    cash = initial_cash
    portfolio_peak = initial_cash
    cooldown_remaining = 0
    history_rows = []

    for i, today in enumerate(all_dates):
        if i == 0:
            history_rows.append({
                "date": today, "symbol": "_TOTAL_",
                "cash": cash, "position": 0, "value": cash,
            })
            continue
        prev = all_dates[i - 1]

        # MDD cap
        if mdd_cap is not None and positions:
            prev_total = cash
            for sym, pos in positions.items():
                df = sym_data[sym]
                if prev in df.index:
                    prev_total += pos["qty"] * float(df.loc[prev, "close"])
            if prev_total > 0 and portfolio_peak > 0:
                dd = (prev_total - portfolio_peak) / portfolio_peak
                if dd <= -mdd_cap:
                    for sym in list(positions.keys()):
                        df = sym_data[sym]
                        if today in df.index:
                            today_open = float(df.loc[today, "open"])
                            if today_open > 0:
                                cash += positions[sym]["qty"] * today_open * (1 - cost_per_side)
                        del positions[sym]
                    cooldown_remaining = cooldown_days

        # 매도 체크 (dual)
        sells = []
        for sym in list(positions.keys()):
            df = sym_data.get(sym)
            if df is None or prev not in df.index or today not in df.index:
                continue
            prev_row = df.loc[prev]
            flow = float(prev_row["flow_ratio"])
            inst = float(prev_row["inst_ratio"])
            if flow < -enter_threshold and inst < -enter_threshold:
                sells.append(sym)
        for sym in sells:
            df = sym_data[sym]
            today_open = float(df.loc[today, "open"])
            if today_open <= 0:
                continue
            cash += positions[sym]["qty"] * today_open * (1 - cost_per_side)
            del positions[sym]

        # 매수 후보
        candidates = []
        if cooldown_remaining <= 0:
            for sym, df in enriched.items():
                if sym in positions:
                    continue
                if prev not in df.index or today not in df.index:
                    continue
                prev_row = df.loc[prev]
                flow = float(prev_row.get("flow_ratio", 0))
                inst = float(prev_row.get("inst_ratio", 0))

                # ML score (있으면)
                ml_score = score_lookup.get((sym, prev), None)

                # 진입 조건
                if use_ml_only:
                    # dual 무시, ML score 만
                    if ml_score is not None and ml_score >= (ml_threshold or 0.5):
                        candidates.append((sym, ml_score, flow, inst))
                else:
                    # dual 통과 필수
                    if flow > enter_threshold and inst > enter_threshold:
                        if ml_threshold is not None:
                            if ml_score is None or ml_score < ml_threshold:
                                continue
                        strength = min(flow, inst)
                        candidates.append((sym, strength, flow, inst))

        # 매수 실행
        free_slots = max_concurrent - len(positions)
        candidates.sort(key=lambda x: -x[1])
        for sym, strength, flow, inst in candidates[:free_slots]:
            df = sym_data[sym]
            today_open = float(df.loc[today, "open"])
            if today_open <= 0:
                continue
            slot_cash = cash / max(1, free_slots)
            px_with_cost = today_open * (1 + cost_per_side)
            qty = int(slot_cash // px_with_cost)
            if qty <= 0:
                continue
            cost_total = qty * today_open * (1 + cost_per_side)
            cash -= cost_total
            positions[sym] = {"qty": qty, "avg_entry": today_open, "entry_date": today}

        # mark
        total_value = cash
        for sym, pos in positions.items():
            df = sym_data[sym]
            if today in df.index:
                total_value += pos["qty"] * float(df.loc[today, "close"])

        if total_value > portfolio_peak:
            portfolio_peak = total_value
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        history_rows.append({
            "date": today, "symbol": "_TOTAL_",
            "cash": cash, "position": len(positions), "value": total_value,
        })

    history = pd.DataFrame(history_rows)
    by_date = history.set_index("date")["value"]
    final = float(by_date.iloc[-1])
    ret = (final - initial_cash) / initial_cash
    peak = by_date.cummax()
    mdd = float(((by_date - peak) / peak).min())
    overall = {
        "initial_cash": initial_cash, "final_value": final,
        "return": ret, "max_drawdown": mdd, "n_days": len(by_date),
    }
    return PortfolioBacktestResult(history, pd.DataFrame(), overall)


def _calmar(ret, mdd, n_days):
    if mdd == 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    return ((1 + ret) ** (1 / years) - 1) / abs(mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "ml_signal_model.pkl"))
    ap.add_argument("--train-split", default="2024-09-01",
                    help="train/test 분할일 (test 만 백테스트)")
    args = ap.parse_args()

    cfg = ForeignFlowBacktestConfig(symbols=DEFAULT_TICKERS, cost_bps=25.0, max_pages=60)

    print(f"=== ML + Dual Combined Backtest ===")
    print(f"train/test split: {args.train_split} (test 기간만 평가)")
    print()

    print("[1/2] ML model 로드...")
    model = load_model(Path(args.model))
    print(f"  model loaded: {args.model}")

    print("\n[2/2] 백테스트 비교...")

    uidx = universe_index(cfg)
    regimes = detect_regimes(uidx, drawdown_threshold=0.05)

    bh = run_buyhold_portfolio(cfg, initial_cash=10_000_000.0)
    bh_test = bh.history.copy()
    bh_test["date"] = pd.to_datetime(bh_test["date"]).dt.normalize()
    bh_test = bh_test[bh_test["date"] >= pd.Timestamp(args.train_split)]
    if not bh_test.empty:
        first_val = bh_test.groupby("date")["value"].sum().iloc[0]
        last_val = bh_test.groupby("date")["value"].sum().iloc[-1]
        bh_ret = (last_val - first_val) / first_val
        # MDD
        td = bh_test.groupby("date")["value"].sum().sort_index()
        bh_mdd = float(((td - td.cummax()) / td.cummax()).min())
        bh_n = len(td)
        bh_calmar = _calmar(bh_ret, bh_mdd, bh_n)
    else:
        bh_ret, bh_mdd, bh_calmar, bh_n = 0, 0, 0, 0

    print(f"\n  {'전략':<35} {'ret':>9} {'MDD':>8} {'Calmar':>8}")
    print(f"  {'-'*60}")
    print(f"  {'BH (test 기간만)':<35} {bh_ret*100:>+8.2f}% {bh_mdd*100:>+7.2f}% {bh_calmar:>+7.2f}")

    # A. Dual only
    res_a = run_combined_backtest(
        cfg, model, test_only=True, train_split_date=args.train_split,
        ml_threshold=None, max_concurrent=7, mdd_cap=0.30,
    )
    o = res_a.overall
    print(f"  {'A. Dual only (current best)':<35} {o['return']*100:>+8.2f}% "
          f"{o['max_drawdown']*100:>+7.2f}% {_calmar(o['return'], o['max_drawdown'], o['n_days']):>+7.2f}")

    # B. Dual + ML threshold
    for th in [0.3, 0.4, 0.5]:
        res_b = run_combined_backtest(
            cfg, model, test_only=True, train_split_date=args.train_split,
            ml_threshold=th, max_concurrent=7, mdd_cap=0.30,
        )
        o = res_b.overall
        print(f"  {'B. Dual + ML ≥ ' + str(th):<35} {o['return']*100:>+8.2f}% "
              f"{o['max_drawdown']*100:>+7.2f}% {_calmar(o['return'], o['max_drawdown'], o['n_days']):>+7.2f}")

    # C. ML only (참고)
    for th in [0.5, 0.6, 0.7]:
        res_c = run_combined_backtest(
            cfg, model, test_only=True, train_split_date=args.train_split,
            ml_threshold=th, use_ml_only=True, max_concurrent=7, mdd_cap=0.30,
        )
        o = res_c.overall
        print(f"  {'C. ML only ≥ ' + str(th):<35} {o['return']*100:>+8.2f}% "
              f"{o['max_drawdown']*100:>+7.2f}% {_calmar(o['return'], o['max_drawdown'], o['n_days']):>+7.2f}")


if __name__ == "__main__":
    main()
