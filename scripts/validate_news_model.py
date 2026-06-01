"""Phase A2-D 검증 — 모델이 진짜 뉴스 알파인가, 단순 momentum 모방인가?

세 실험으로 분리:
  Exp 1  Random (no-news) baseline — 뉴스 없는 random 5분봉 시점에서 동일
         pre_news_features + label → R² 비교. 비슷하면 momentum 만 학습.
  Exp 2  Walk-forward — 학습/테스트 cutoff 를 시간 순서로 이동시키며 R²
         일관성 확인. 음수 나오면 overfit.
  Exp 3  News-content only — pre-news momentum/vol 제거, sentiment/event/시간
         만으로 학습. R² ≈ 0 이면 뉴스 자체엔 정보 거의 없음.

Output: 표 + JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import random
from datetime import datetime, timedelta
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

from autotrader.data.minute_bars import fetch_minute_bars, get_bar_at_or_after
from autotrader.news_signal.features import pre_news_features
from autotrader.news_signal.historical import (
    load_historical, add_historical_features,
    HIST_BASE_FEATURES, HIST_DPLUS_FEATURES, HIST_FEATURE_COLS, HIST_LABEL_COL,
    train_historical,
)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))


def _r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ============================================================ Exp 1 — random

def build_random_dataset(tickers: list[str], n_per_ticker: int = 300,
                          seed: int = 42) -> pd.DataFrame:
    """뉴스 없는 random 5분봉 시점 sampling → pre_news_features + label.

    각 ticker 의 minute bars 에서 random 시점 N개 추출:
      - 장중 (9:30 ~ 14:00 KST, 첫 30분 / 마지막 90분 회피)
      - 같은 거래일 내 entry/exit 가능
    """
    rng = random.Random(seed)
    rows = []
    for ticker in tickers:
        try:
            bars = fetch_minute_bars(ticker, period="60d", interval="5m")
        except Exception as e:
            print(f"  [{ticker}] bars fetch fail: {e}")
            continue
        if bars.empty:
            continue
        # 후보 timestamps
        candidates = []
        for ts in bars.index:
            if ts.hour < 9 or ts.hour > 14:
                continue
            if ts.hour == 9 and ts.minute < 30:
                continue
            if ts.hour == 14 and ts.minute > 0:
                continue
            if ts.weekday() >= 5:
                continue
            candidates.append(ts)
        if len(candidates) < 50:
            continue
        sample_n = min(n_per_ticker, len(candidates))
        chosen = rng.sample(candidates, sample_n)

        for t_pseudo in chosen:
            pre = pre_news_features(bars, t_pseudo, lookback_bars=18)
            if pre.pre_n_bars < 6:
                continue
            t_entry = t_pseudo + pd.Timedelta(minutes=5)
            t_exit = t_pseudo + pd.Timedelta(minutes=65)
            entry_bar = get_bar_at_or_after(bars, t_entry.to_pydatetime())
            exit_bar = get_bar_at_or_after(bars, t_exit.to_pydatetime())
            if entry_bar is None or exit_bar is None:
                continue
            if entry_bar.name >= exit_bar.name:
                continue
            if entry_bar.name.date() != exit_bar.name.date():
                continue
            entry_p = float(entry_bar["Close"])
            exit_p = float(exit_bar["Close"])
            if entry_p <= 0 or exit_p <= 0:
                continue
            gross = (exit_p - entry_p) / entry_p
            rows.append({
                "ticker": ticker,
                "t_news": t_pseudo,
                "event": "neutral",
                "confidence": 1.0,
                "gross_pnl": gross,
                "sent_score": 0.0,
                "n_pos_kw": 0,
                "n_neg_kw": 0,
                "momentum_5m": pre.momentum_5m,
                "momentum_15m": pre.momentum_15m,
                "momentum_30m": pre.momentum_30m,
                "vol_30m": pre.vol_30m,
                "volume_z": pre.volume_z,
                "pre_n_bars": pre.pre_n_bars,
            })
    return pd.DataFrame(rows)


def exp1_random_baseline(news_df: pd.DataFrame) -> dict:
    """뉴스 없는 random 시점에서 같은 features 로 학습."""
    print("\n=== Exp 1 — Random (no-news) baseline ===")
    tickers = sorted(news_df["ticker"].unique())
    print(f"  sampling {len(tickers)} tickers, ~300 per ticker...")
    rdf = build_random_dataset(tickers, n_per_ticker=300)
    if rdf.empty:
        return {"error": "empty random dataset"}
    print(f"  built {len(rdf)} random samples")

    # 같은 학습 파이프라인
    rdf_feat = add_historical_features(rdf)
    res = train_historical(rdf_feat, test_frac=0.2, n_estimators=300)
    print(f"  train n={res.train_size} test n={res.test_size}")
    print(f"  train R²={res.train_r2:+.4f} test R²={res.test_r2:+.4f}")
    print(f"  test RMSE={res.test_rmse:.6f} vs baseline RMSE={res.baseline_test_rmse:.6f}")
    return {
        "n_total": len(rdf),
        "train_r2": res.train_r2,
        "test_r2": res.test_r2,
        "test_rmse": res.test_rmse,
        "baseline_rmse": res.baseline_test_rmse,
        "rmse_lift_pct": (res.baseline_test_rmse - res.test_rmse) / res.baseline_test_rmse * 100,
    }


# ============================================================ Exp 2 — walk-forward

def exp2_walk_forward(news_df: pd.DataFrame) -> list[dict]:
    """시간 순서 cutoff 여러 개로 학습/테스트 R² 일관성 확인."""
    print("\n=== Exp 2 — Walk-forward validation ===")
    df = news_df.sort_values("t_news").reset_index(drop=True)
    df = add_historical_features(df)

    # 4월 데이터 → 주별 cutoff
    cutoffs = [
        ("2026-04-13", "2026-04-20"),  # train: 3/30~4/13, test: 4/13~4/20
        ("2026-04-20", "2026-04-27"),  # train: 3/30~4/20, test: 4/20~4/27
    ]
    results = []
    for train_end, test_end in cutoffs:
        train_end_ts = pd.Timestamp(train_end)
        test_end_ts = pd.Timestamp(test_end)
        train = df[df["t_news"] < train_end_ts]
        test = df[(df["t_news"] >= train_end_ts) & (df["t_news"] < test_end_ts)]
        if len(train) < 100 or len(test) < 50:
            print(f"  [{train_end} → {test_end}] skip (too small)")
            continue

        try:
            import lightgbm as lgb
        except ImportError:
            return [{"error": "lightgbm missing"}]

        # 라벨/feature 채움
        train = train.dropna(subset=[HIST_LABEL_COL]).copy()
        test = test.dropna(subset=[HIST_LABEL_COL]).copy()
        for col in HIST_FEATURE_COLS:
            if col not in train.columns: train[col] = 0.0
            if col not in test.columns: test[col] = 0.0
        train[HIST_FEATURE_COLS] = train[HIST_FEATURE_COLS].fillna(0.0)
        test[HIST_FEATURE_COLS] = test[HIST_FEATURE_COLS].fillna(0.0)

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
             "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "seed": 42},
            train_ds, num_boost_round=300,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                       lgb.log_evaluation(0)],
        )
        pred = model.predict(X_test)
        baseline_rmse = _rmse(y_test, np.zeros_like(y_test.values))
        r2 = _r2(y_test, pred)
        rmse = _rmse(y_test, pred)
        lift = (baseline_rmse - rmse) / baseline_rmse * 100
        print(f"  [train<{train_end} → test<{test_end}] "
              f"n_train={len(train)} n_test={len(test)} "
              f"R²={r2:+.4f} RMSE={rmse:.6f} (vs baseline {baseline_rmse:.6f}, lift {lift:+.2f}%)")
        results.append({
            "train_end": train_end, "test_end": test_end,
            "n_train": len(train), "n_test": len(test),
            "test_r2": r2, "test_rmse": rmse,
            "baseline_rmse": baseline_rmse, "rmse_lift_pct": lift,
        })
    return results


# ============================================================ Exp 3 — news-only

def exp3_news_only(news_df: pd.DataFrame) -> dict:
    """Pre-news momentum/vol 제거. sentiment/event/시간 만으로 학습."""
    print("\n=== Exp 3 — News-content only (no pre-news momentum) ===")

    df = add_historical_features(news_df)
    df = df.dropna(subset=[HIST_LABEL_COL]).copy()
    for col in HIST_FEATURE_COLS:
        if col not in df.columns: df[col] = 0.0
    df[HIST_FEATURE_COLS] = df[HIST_FEATURE_COLS].fillna(0.0)
    df = df.sort_values("t_news").reset_index(drop=True)

    # Pre-news features 만 제외
    drop_cols = ["momentum_5m", "momentum_15m", "momentum_30m",
                 "vol_30m", "volume_z", "pre_n_bars"]
    use_cols = [c for c in HIST_FEATURE_COLS if c not in drop_cols]
    print(f"  features used: {use_cols}")

    n = len(df)
    n_test = int(n * 0.2)
    n_train = n - n_test
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]

    try:
        import lightgbm as lgb
    except ImportError:
        return {"error": "lightgbm missing"}

    X_train = train[use_cols].astype(float)
    y_train = train[HIST_LABEL_COL].astype(float)
    X_test = test[use_cols].astype(float)
    y_test = test[HIST_LABEL_COL].astype(float)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)
    model = lgb.train(
        {"objective": "regression", "metric": "rmse",
         "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31,
         "min_data_in_leaf": 20, "feature_fraction": 0.9,
         "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "seed": 42},
        train_ds, num_boost_round=300,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(0)],
    )
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    baseline_rmse = _rmse(y_test, np.zeros_like(y_test.values))
    r2_train = _r2(y_train, pred_train)
    r2_test = _r2(y_test, pred_test)
    rmse_test = _rmse(y_test, pred_test)
    lift = (baseline_rmse - rmse_test) / baseline_rmse * 100

    print(f"  n_train={n_train} n_test={n_test}")
    print(f"  train R²={r2_train:+.4f} test R²={r2_test:+.4f}")
    print(f"  test RMSE={rmse_test:.6f} vs baseline RMSE={baseline_rmse:.6f} (lift {lift:+.2f}%)")
    importance = dict(sorted(zip(use_cols, model.feature_importance(
        importance_type="gain").tolist()), key=lambda kv: -kv[1]))
    return {
        "n_train": n_train, "n_test": n_test,
        "train_r2": r2_train, "test_r2": r2_test,
        "test_rmse": rmse_test, "baseline_rmse": baseline_rmse,
        "rmse_lift_pct": lift, "feature_importance": importance,
    }


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "data" / "validate_news_model.json"))
    args = ap.parse_args()

    print("=== Loading historical data ===")
    df = load_historical(Path(args.parquet_dir))
    print(f"  rows: {len(df)}")
    print(f"  date range: {df.t_news.min()} → {df.t_news.max()}")
    print(f"  tickers: {sorted(df.ticker.unique())}")

    out = {
        "input_rows": len(df),
        "input_tickers": sorted(df.ticker.unique()),
    }

    out["exp1_random_baseline"] = exp1_random_baseline(df)
    out["exp2_walk_forward"] = exp2_walk_forward(df)
    out["exp3_news_only"] = exp3_news_only(df)

    Path(args.out).write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n→ saved: {args.out}")


if __name__ == "__main__":
    main()
