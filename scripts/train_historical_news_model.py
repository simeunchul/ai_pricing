"""historical news → 1시간 stock return 예측 모델 학습.

기존 backtest_news_intraday parquet 12K rows 활용. 새 KIS 데이터 안 모아도
바로 학습 가능.

Usage:
  python scripts/train_historical_news_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from autotrader.news_signal.historical import (
    load_historical, add_historical_features, train_historical,
    HIST_FEATURE_COLS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default=str(ROOT / "data"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--save",
                    default=str(ROOT / "data" / "fair_model_historical.lgbm"))
    args = ap.parse_args()

    print("=== Phase A2-historical — LGBM on real backtest parquets ===")
    df = load_historical(Path(args.parquet_dir))
    print(f"  date range: {df.t_news.min()} → {df.t_news.max()}")
    print(f"  tickers   : {sorted(df.ticker.unique())}")
    print(f"  events    : {df.event.value_counts().to_dict()}")

    df = add_historical_features(df)
    print(f"  features  : {HIST_FEATURE_COLS}")
    print(f"  label     : gross_pnl  (1-hour stock return)")
    print()

    print("=== train ===")
    res = train_historical(
        df, test_frac=args.test_frac,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
    )
    print(f"  train n   : {res.train_size}")
    print(f"  test  n   : {res.test_size}")
    print(f"  train RMSE: {res.train_rmse:.6f}")
    print(f"  test  RMSE: {res.test_rmse:.6f}")
    print(f"  train MAE : {res.train_mae:.6f}")
    print(f"  test  MAE : {res.test_mae:.6f}")
    print(f"  train R²  : {res.train_r2:+.4f}")
    print(f"  test  R²  : {res.test_r2:+.4f}")
    print()

    print(f"=== baseline (always-zero) ===")
    print(f"  test RMSE : {res.baseline_test_rmse:.6f}")
    rmse_lift = (res.baseline_test_rmse - res.test_rmse) / res.baseline_test_rmse * 100
    print(f"  → LightGBM RMSE 개선: {rmse_lift:+.2f}% vs baseline")
    if res.test_r2 > 0:
        print(f"  → ✓ R² 양수 = 모델이 일반화 — 봇 hook 후보")
    elif res.test_rmse < res.baseline_test_rmse:
        print(f"  → △ baseline 보다 미세하게 개선")
    else:
        print(f"  → ✗ baseline 못 이김 — 추가 데이터/feature 필요")
    print()

    print("=== feature importance (gain) ===")
    for i, (k, v) in enumerate(list(res.feature_importance.items()), 1):
        print(f"  {i:>2}. {k:<22} {v:>10.1f}")
    print()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    res.model.save_model(str(save_path))
    print(f"→ model saved: {save_path}")

    pred_path = save_path.with_suffix(".predictions.csv")
    res.test_predictions.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"→ test predictions: {pred_path}")

    metrics_path = save_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "n_train": res.train_size, "n_test": res.test_size,
        "train_rmse": res.train_rmse, "test_rmse": res.test_rmse,
        "train_mae": res.train_mae, "test_mae": res.test_mae,
        "train_r2": res.train_r2, "test_r2": res.test_r2,
        "baseline_test_rmse": res.baseline_test_rmse,
        "rmse_lift_vs_baseline_pct": rmse_lift,
        "feature_importance": res.feature_importance,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ metrics: {metrics_path}")


if __name__ == "__main__":
    main()
