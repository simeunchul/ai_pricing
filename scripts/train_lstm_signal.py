"""LSTM 시계열 모델 학습 — 종목별 symbol embedding + sequence learning.

LightGBM (cross-sectional) 의 한계 보완:
  - 시간 순서 의존성 자동 학습 (lag features 수동 X)
  - Symbol embedding 으로 종목별 특성 학습

Usage:
  python scripts/train_lstm_signal.py
  python scripts/train_lstm_signal.py --epochs 20 --train-split 2024-09-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

import pandas as pd

from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig, _load_symbol_data,
)
from autotrader.ml import build_features
from autotrader.ml.lstm_signal_model import (
    train_lstm, save_lstm, predict_lstm,
)

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
    ap.add_argument("--train-split", default="2024-09-01")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--out-model", default=str(ROOT / "data" / "lstm_signal_model.pt"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    cfg = ForeignFlowBacktestConfig(symbols=tickers, cost_bps=25.0, max_pages=60)

    print(f"=== LSTM Signal Training ===")
    print(f"tickers: {len(tickers)}")
    print(f"train/test split: {args.train_split}")
    print(f"seq_len={args.seq_len}, epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print()

    print("[1/3] 데이터 로드...")
    sym_data = {sym: _load_symbol_data(sym, cfg) for sym in tickers}
    sym_data = {s: d for s, d in sym_data.items() if not d.empty}
    print(f"  {len(sym_data)} 종목")

    print("\n[2/3] Features build...")
    df = build_features(sym_data)
    print(f"  features rows: {len(df)}")
    print(f"  symbols: {df['symbol'].nunique()}")
    print(f"  target balance: {df['target'].mean()*100:.2f}% positive")

    print("\n[3/3] LSTM 학습...")
    result = train_lstm(
        df,
        train_split_date=args.train_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_len=args.seq_len,
    )

    print(f"\n=== 학습 결과 ===")
    print(f"  train sequences: {result.n_train:,}")
    print(f"  test sequences : {result.n_test:,}")
    print(f"  best train AUC : {result.train_auc:.4f}")
    print(f"  final test AUC : {result.test_auc:.4f}")
    print(f"  test accuracy  : {result.test_accuracy*100:.2f}%")

    save_lstm(result, Path(args.out_model))
    print(f"\n[saved] {args.out_model}")

    # 비교용: LightGBM AUC 와 직접 비교
    print(f"\n=== 비교 (이전 LightGBM 결과) ===")
    print(f"  LightGBM test AUC: 0.5307 (cross-sectional)")
    print(f"  LSTM     test AUC: {result.test_auc:.4f} (per-symbol + sequence)")
    if result.test_auc > 0.5307:
        print(f"  → LSTM 이 +{(result.test_auc - 0.5307)*100:.2f}%p 더 좋음")
    else:
        print(f"  → LightGBM 이 더 좋음 (LSTM overfit 또는 데이터 부족)")


if __name__ == "__main__":
    main()
