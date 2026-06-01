"""Dense walk-forward — 30일 데이터 내에서 매주 cutoff 굴려 R² 분포 확인.

Phase A2-D 검증 1차에선 cutoff 2개만 했음 (둘 다 음수). 더 많은 cutoff 로
음수가 일관적인지, 우연인지 확인.

Output: cutoff 별 (n_train, n_test, R², lift) 테이블.
"""

from __future__ import annotations

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

from autotrader.news_signal.historical import (
    load_historical, add_historical_features,
    HIST_FEATURE_COLS, HIST_LABEL_COL,
)


def _r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))


def main():
    print("=== Dense walk-forward validation ===")
    df = load_historical(Path("data"))
    df = add_historical_features(df)
    df = df.dropna(subset=[HIST_LABEL_COL]).copy()
    for col in HIST_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[HIST_FEATURE_COLS] = df[HIST_FEATURE_COLS].fillna(0.0)
    df = df.sort_values("t_news").reset_index(drop=True)

    print(f"  rows: {len(df)}, span: {df.t_news.min()} → {df.t_news.max()}")

    # 4월 5일부터 매일 cutoff (test window 7일)
    import lightgbm as lgb

    cutoffs = pd.date_range("2026-04-05", "2026-04-21", freq="3D")
    print(f"\n  cutoffs: {[c.date().isoformat() for c in cutoffs]}")
    print(f"\n  {'cutoff':<12} {'n_tr':>5} {'n_te':>5} {'test_R²':>9} {'lift%':>7}")
    print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*9} {'-'*7}")

    results = []
    for cut in cutoffs:
        train_end = cut
        test_end = cut + pd.Timedelta(days=7)
        train = df[df["t_news"] < train_end]
        test = df[(df["t_news"] >= train_end) & (df["t_news"] < test_end)]
        if len(train) < 100 or len(test) < 30:
            print(f"  {cut.date().isoformat():<12} {len(train):>5} {len(test):>5}   (skip — too small)")
            continue

        X_train = train[HIST_FEATURE_COLS].astype(float)
        y_train = train[HIST_LABEL_COL].astype(float)
        X_test = test[HIST_FEATURE_COLS].astype(float)
        y_test = test[HIST_LABEL_COL].astype(float)

        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)
        model = lgb.train(
            {"objective": "regression", "metric": "rmse",
             "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31,
             "min_data_in_leaf": 20, "feature_fraction": 0.9,
             "bagging_fraction": 0.8, "bagging_freq": 5,
             "verbose": -1, "seed": 42},
            train_ds, num_boost_round=300, valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred = model.predict(X_test)
        baseline_rmse = _rmse(y_test, np.zeros_like(y_test.values))
        rmse = _rmse(y_test, pred)
        r2 = _r2(y_test, pred)
        lift = (baseline_rmse - rmse) / baseline_rmse * 100
        print(f"  {cut.date().isoformat():<12} {len(train):>5} {len(test):>5} "
              f"{r2:>+9.4f} {lift:>+6.2f}%")
        results.append({
            "cutoff": cut.date().isoformat(),
            "n_train": len(train), "n_test": len(test),
            "test_r2": r2, "rmse_lift_pct": lift,
        })

    if results:
        r2s = [r["test_r2"] for r in results]
        lifts = [r["rmse_lift_pct"] for r in results]
        print(f"\n  === Summary ({len(results)} cutoffs) ===")
        print(f"  R² mean   = {np.mean(r2s):+.4f}")
        print(f"  R² std    = {np.std(r2s):.4f}")
        print(f"  R² min    = {min(r2s):+.4f}")
        print(f"  R² max    = {max(r2s):+.4f}")
        print(f"  R² > 0    : {sum(1 for r in r2s if r > 0)}/{len(r2s)}")
        print(f"  lift mean = {np.mean(lifts):+.2f}%")

    Path("data/walk_forward_dense.json").write_text(
        json.dumps({"results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\n→ saved: data/walk_forward_dense.json")


if __name__ == "__main__":
    main()
