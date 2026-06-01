"""Phase A2 — fair-value LightGBM 학습 스크립트.

Usage:
  python scripts/train_news_signal.py
  python scripts/train_news_signal.py --logs data/kis_trading_log_*.json
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

from autotrader.news_signal.etl import ETL_Config, build_features
from autotrader.news_signal.model import train_lightgbm, baseline_naive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+",
                    default=[str(ROOT / "data" / "kis_trading_log_20260428.json")])
    ap.add_argument("--news-dir", default=str(ROOT / "data" / "news_cache"))
    ap.add_argument("--bucket-min", type=int, default=5)
    ap.add_argument("--classifier", default="rule", choices=["rule", "finbert"])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--save", default=str(ROOT / "data" / "fair_model.lgbm"))
    args = ap.parse_args()

    log_paths = [Path(p) for p in args.logs]
    print(f"=== Phase A2 — fair-value model train ===")
    print(f"  tick logs: {len(log_paths)}")
    for p in log_paths:
        print(f"    - {p.name}")
    print(f"  news dir : {args.news_dir}")
    print(f"  bucket   : {args.bucket_min}min")
    print(f"  classifier: {args.classifier}")
    print()

    # ETL
    cfg = ETL_Config(
        tick_log_paths=log_paths,
        news_dir=Path(args.news_dir),
        bucket_min=args.bucket_min,
        classifier=args.classifier,
    )
    df = build_features(cfg)
    if df.empty:
        print("ETL produced empty dataset. abort."); sys.exit(1)
    print(f"ETL → {len(df)} rows × {len(df.columns)} cols")
    print(f"  symbols     : {sorted(df.symbol.unique())}")
    print(f"  bucket span : {df.ts_bucket.min()} → {df.ts_bucket.max()}")
    print(f"  with-news   : {(df.news_count_total > 0).sum()} ({(df.news_count_total > 0).mean()*100:.1f}%)")
    print()

    # Train
    print("=== train ===")
    res = train_lightgbm(
        df, test_frac=args.test_frac, n_estimators=args.n_estimators,
    )
    print(f"  train n   : {res.train_size}")
    print(f"  test  n   : {res.test_size}")
    print(f"  train RMSE: {res.train_rmse:.6f} (= {res.train_rmse*100:.3f}%)")
    print(f"  test  RMSE: {res.test_rmse:.6f} (= {res.test_rmse*100:.3f}%)")
    print(f"  train MAE : {res.train_mae:.6f}")
    print(f"  test  MAE : {res.test_mae:.6f}")
    print(f"  train R²  : {res.train_r2:+.4f}")
    print(f"  test  R²  : {res.test_r2:+.4f}")
    print()

    # Baseline 비교
    base = baseline_naive(df, test_frac=args.test_frac)
    print(f"=== baseline (always-zero) ===")
    print(f"  test RMSE : {base['test_rmse']:.6f}")
    print(f"  test MAE  : {base['test_mae']:.6f}")
    print(f"  test R²   : {base['test_r2']:+.4f}")
    rmse_lift = (base['test_rmse'] - res.test_rmse) / base['test_rmse'] * 100
    print(f"  → LightGBM RMSE 개선: {rmse_lift:+.2f}% vs baseline")
    if res.test_r2 > 0:
        print(f"  → R² 양수 = 모델이 일부 정보 학습 ✓")
    else:
        print(f"  → R² 음수 = baseline 보다 나쁨 (overfit 또는 데이터 부족)")
    print()

    # Feature importance (top 10)
    print("=== feature importance (gain) ===")
    for i, (k, v) in enumerate(list(res.feature_importance.items())[:10], 1):
        print(f"  {i:>2}. {k:<28} {v:>10.1f}")
    print()

    # 모델 저장
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    res.model.save_model(str(save_path))
    print(f"→ model saved: {save_path}")

    # test predictions 저장 (분석용)
    pred_path = save_path.with_suffix(".predictions.csv")
    res.test_predictions.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"→ test predictions: {pred_path}")

    # metrics JSON 저장
    metrics_path = save_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "n_train": res.train_size,
        "n_test": res.test_size,
        "train_rmse": res.train_rmse, "test_rmse": res.test_rmse,
        "train_mae": res.train_mae, "test_mae": res.test_mae,
        "train_r2": res.train_r2, "test_r2": res.test_r2,
        "baseline_test_rmse": base["test_rmse"],
        "baseline_test_mae": base["test_mae"],
        "rmse_lift_vs_baseline_pct": rmse_lift,
        "feature_importance": res.feature_importance,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ metrics: {metrics_path}")


if __name__ == "__main__":
    main()
