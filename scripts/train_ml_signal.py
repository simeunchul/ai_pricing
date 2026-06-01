"""LightGBM 학습 — 외국인+기관 + 차트 features → 다음날 수익률 예측.

40 종목 × 1200 거래일 데이터로 학습.

Usage:
  python scripts/train_ml_signal.py
  python scripts/train_ml_signal.py --train-split 2025-06-01    # 명시적 split
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
    ForeignFlowBacktestConfig, _load_symbol_data,
)
from autotrader.ml import build_features, train_lightgbm, save_model

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--train-split", default=None,
                    help="train/test 분할 날짜 (YYYY-MM-DD), 미지정 시 80% 시간순")
    ap.add_argument("--out-model", default=str(ROOT / "data" / "ml_signal_model.pkl"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(symbols=tickers, cost_bps=25.0, max_pages=60)

    print(f"=== Dual Signal ML Training ({len(tickers)} 종) ===")
    print(f"train/test split: {args.train_split or 'auto 80%'}")
    print()

    print("[1/4] 데이터 로드...")
    sym_data = {sym: _load_symbol_data(sym, cfg) for sym in tickers}
    sym_data = {s: d for s, d in sym_data.items() if not d.empty}
    total_days = sum(len(d) for d in sym_data.values())
    print(f"  {len(sym_data)} 종목, 총 {total_days} 종목-일")

    print("\n[2/4] Features 빌드 (차트 + ML features + target)...")
    df = build_features(sym_data)
    print(f"  features rows: {len(df)} (NaN drop 후)")
    print(f"  target balance: {df['target'].mean()*100:.2f}% positive (T+1 ≥ +0.5%)")

    print("\n[3/4] LightGBM 학습...")
    result = train_lightgbm(df, train_split_date=args.train_split, train_frac=0.8)

    print(f"\n=== 학습 결과 ===")
    print(f"  train samples: {result.n_train:,}  test: {result.n_test:,}")
    print(f"  train AUC : {result.train_auc:.4f}")
    print(f"  test AUC  : {result.test_auc:.4f}")
    print(f"  test acc  : {result.test_accuracy*100:.2f}%")
    print(f"  test prec : {result.test_precision*100:.2f}%  (양성 예측의 정확도)")

    print(f"\n=== Feature Importance (gain) ===")
    sorted_imp = sorted(result.feature_importance.items(), key=lambda x: -x[1])
    for name, gain in sorted_imp:
        print(f"  {name:<22} {gain:>10.0f}")

    # save
    save_model(result.model, Path(args.out_model))
    print(f"\n[saved] model: {args.out_model}")

    # save predictions
    pred_path = Path(args.out_model).parent / "ml_test_predictions.parquet"
    result.test_predictions.to_parquet(pred_path)
    print(f"[saved] test predictions: {pred_path}")


if __name__ == "__main__":
    main()
