"""Phase A2-D / C — 6개월 1h-bar + DART events 학습 + walk-forward 검증.

Usage:
  python scripts/train_v2_dart_1h.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from autotrader.news_signal.historical_v2 import (
    ETLv2Config, build_dataset_v2,
    FEATURE_COLS_V2, LABEL_COL_V2, TICKER_TO_CORP_CODE,
)


def _r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def train_one(train, test):
    import lightgbm as lgb
    X_train = train[FEATURE_COLS_V2].astype(float)
    y_train = train[LABEL_COL_V2].astype(float)
    X_test = test[FEATURE_COLS_V2].astype(float)
    y_test = test[LABEL_COL_V2].astype(float)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)
    model = lgb.train(
        {"objective": "regression", "metric": "rmse",
         "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31,
         "min_data_in_leaf": 10, "feature_fraction": 0.9,
         "bagging_fraction": 0.8, "bagging_freq": 5,
         "verbose": -1, "seed": 42},
        train_ds, num_boost_round=300, valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(0)],
    )
    pred = model.predict(X_test)
    baseline_rmse = _rmse(y_test, np.zeros_like(y_test.values))
    return {
        "model": model,
        "test_r2": _r2(y_test, pred),
        "test_rmse": _rmse(y_test, pred),
        "baseline_rmse": baseline_rmse,
        "rmse_lift_pct": (baseline_rmse - _rmse(y_test, pred)) / baseline_rmse * 100,
        "feature_importance": dict(sorted(zip(
            FEATURE_COLS_V2,
            model.feature_importance(importance_type="gain").tolist(),
        ), key=lambda kv: -kv[1])),
        "n_train": len(train), "n_test": len(test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--save", default=str(ROOT / "data" / "fair_model_v2_dart_1h.json"))
    args = ap.parse_args()

    cfg = ETLv2Config(
        tickers=list(TICKER_TO_CORP_CODE.keys()),
        days_history=args.days,
        bars_period=f"{args.days}d",
        bars_interval="1h",
    )
    df = build_dataset_v2(cfg)
    if df.empty:
        print("Empty dataset"); sys.exit(1)
    print(f"\nDataset: {len(df)} rows")
    print(f"  span: {df.t_event.min()} → {df.t_event.max()}")
    print(f"  tickers: {sorted(df.ticker.unique())}")
    print(f"  events:")
    print(df.event.value_counts().head(15).to_string())
    print()

    # 시간 정렬
    df = df.sort_values("t_event").reset_index(drop=True)

    # walk-forward — 1주 단위 cutoff (test window 4주)
    span_start = df.t_event.min()
    span_end = df.t_event.max()
    print(f"=== Walk-forward (4주 test window, 1주 step) ===")
    print(f"  {'cutoff':<12} {'n_tr':>5} {'n_te':>5} {'test_R²':>9} {'lift%':>7}")
    print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*9} {'-'*7}")

    cutoffs = pd.date_range(
        span_start + pd.Timedelta(days=60), span_end - pd.Timedelta(days=14),
        freq="14D", tz=span_start.tz,
    )

    wf_results = []
    for cut in cutoffs:
        train = df[df["t_event"] < cut]
        test = df[(df["t_event"] >= cut)
                  & (df["t_event"] < cut + pd.Timedelta(days=28))]
        if len(train) < 100 or len(test) < 30:
            continue
        try:
            r = train_one(train, test)
        except Exception as e:
            print(f"  {cut.date()} train fail: {e}")
            continue
        print(f"  {cut.date().isoformat():<12} {r['n_train']:>5} {r['n_test']:>5} "
              f"{r['test_r2']:>+9.4f} {r['rmse_lift_pct']:>+6.2f}%")
        wf_results.append({
            "cutoff": cut.date().isoformat(),
            "n_train": r["n_train"], "n_test": r["n_test"],
            "test_r2": r["test_r2"], "rmse_lift_pct": r["rmse_lift_pct"],
        })

    if wf_results:
        r2s = [r["test_r2"] for r in wf_results]
        lifts = [r["rmse_lift_pct"] for r in wf_results]
        print(f"\n  === Summary ({len(wf_results)} cutoffs) ===")
        print(f"  R² mean   = {np.mean(r2s):+.4f}")
        print(f"  R² std    = {np.std(r2s):.4f}")
        print(f"  R² min    = {min(r2s):+.4f}")
        print(f"  R² max    = {max(r2s):+.4f}")
        print(f"  R² > 0    : {sum(1 for r in r2s if r > 0)}/{len(r2s)}")
        print(f"  lift mean = {np.mean(lifts):+.2f}%")

    # full 80/20 split for feature importance
    n_test = max(50, int(len(df) * 0.2))
    final = train_one(df.iloc[:-n_test], df.iloc[-n_test:])
    print(f"\n=== Full 80/20 split (final model) ===")
    print(f"  test R² = {final['test_r2']:+.4f}")
    print(f"  RMSE lift vs baseline = {final['rmse_lift_pct']:+.2f}%")
    print(f"\n  feature importance (top 8):")
    for k, v in list(final["feature_importance"].items())[:8]:
        print(f"    {k:<22} {v:>10.1f}")

    # save
    out = {
        "n_total": len(df),
        "tickers": sorted(df.ticker.unique()),
        "wf_results": wf_results,
        "final": {k: v for k, v in final.items() if k != "model"},
    }
    Path(args.save).write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n→ saved: {args.save}")


if __name__ == "__main__":
    main()
