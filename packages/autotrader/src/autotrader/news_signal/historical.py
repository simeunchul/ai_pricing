"""Historical training pipeline — backtest_news_intraday parquet 활용.

기존 6개 parquet 파일 (~12,000 labeled rows, 2026-03-30~04-24) 을 합쳐서
stock-level fair-value 모델 학습. 이건 "ETF 차익 봇 1주일 운영하고 학습" 의
훨씬 빠른 대안.

각 row = (ticker, t_news, event, confidence, entry_price, exit_price, ...)
  - 뉴스 발생 → T+5min 진입 → T+65min 청산 시뮬레이션
  - label: gross_pnl = (exit - entry) / entry  (1시간 stock return)
  - feature: event/confidence/ticker + 시간 (hour, weekday)

추론 시 (봇 hook): 새 뉴스 들어오면 모델로 1시간 return 예측 →
ETF 비중 가중 → ETF fair-value direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# stock-level features (ETL 모듈과 분리 — 이건 이벤트 단위)
HIST_BASE_FEATURES = [
    "confidence",
    "event_id",
    "ticker_id",
    "hour",
    "minute",
    "weekday",
    "minutes_since_open",  # 09:00 KST 기준
]

# Phase A2-D 추가 features (parquet 에 새로 들어옴)
HIST_DPLUS_FEATURES = [
    "sent_score",          # -1 ~ +1 강화 sentiment lexicon
    "n_pos_kw",            # positive 키워드 적중 수
    "n_neg_kw",            # negative 키워드 적중 수
    "momentum_5m",         # t_news 직전 5분 가격 변화율
    "momentum_15m",        # 직전 15분
    "momentum_30m",        # 직전 30분
    "vol_30m",             # 직전 30분 변동성
    "volume_z",            # 직전 거래량 z-score
    "pre_n_bars",          # 사용 가능 bars (data quality)
]

HIST_FEATURE_COLS = HIST_BASE_FEATURES + HIST_DPLUS_FEATURES
HIST_LABEL_COL = "gross_pnl"


# 이벤트 카테고리 → numeric (rule classifier 의 출력에 맞춤)
EVENT_CATEGORIES = [
    "neutral", "earnings_beat", "earnings_miss",
    "mna", "rating_change", "regulatory", "macro_shock",
    "other",
]


def load_historical(parquet_dir: Path,
                     pattern: str = "backtest_news_intraday_*.parquet"
                     ) -> pd.DataFrame:
    """모든 parquet 합치고 (ticker, t_news, title) 기준 dedupe.

    Returns DataFrame with raw columns plus parsed timestamps.
    """
    files = sorted(parquet_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet matching {pattern} in {parquet_dir}")

    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  parquet load fail {f.name}: {e}")
    df = pd.concat(parts, ignore_index=True)

    # dedupe — 같은 뉴스에 대해 여러 번 backtest 했을 수 있음.
    # keep="last" → 가장 최근 backtest (= 새 features 포함 가능성 높음) 우선.
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "t_news", "title"], keep="last")
    after = len(df)
    print(f"  loaded {before} rows, deduped to {after}")

    # parse timestamps
    for col in ("t_news", "t_entry", "t_exit"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            # KST 변환 (보고용; 시간 feature 추출에 필요)
            df[col] = df[col].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)

    return df


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """row-level features for LGBM."""
    df = df.copy()
    df["event"] = df["event"].fillna("other")
    df["event_id"] = df["event"].apply(
        lambda x: EVENT_CATEGORIES.index(x) if x in EVENT_CATEGORIES else len(EVENT_CATEGORIES) - 1
    ).astype(float)
    df["ticker_id"] = pd.Categorical(df["ticker"]).codes.astype(float)
    df["hour"] = df["t_news"].dt.hour.astype(float)
    df["minute"] = df["t_news"].dt.minute.astype(float)
    df["weekday"] = df["t_news"].dt.weekday.astype(float)
    df["minutes_since_open"] = ((df["hour"] - 9) * 60 + df["minute"]).astype(float)
    return df


@dataclass
class HistoricalTrainResult:
    model: object
    feature_cols: list[str]
    train_size: int
    test_size: int
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    baseline_test_rmse: float
    feature_importance: dict[str, float]
    test_predictions: pd.DataFrame


def _rmse(y_true, y_pred): return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))
def _mae(y_true, y_pred): return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
def _r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def train_historical(df: pd.DataFrame,
                      test_frac: float = 0.2,
                      n_estimators: int = 300,
                      learning_rate: float = 0.05,
                      max_depth: int = 5,
                      seed: int = 42) -> HistoricalTrainResult:
    """Time-ordered split + LGBM. df 는 add_historical_features 통과 가정."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError("pip install lightgbm") from e

    # 라벨은 필수, features 는 옛 parquet 에 신규 컬럼 없을 수 있으므로 0 채움
    df = df.dropna(subset=[HIST_LABEL_COL]).copy()
    for col in HIST_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[HIST_FEATURE_COLS] = df[HIST_FEATURE_COLS].fillna(0.0)
    df = df.sort_values("t_news").reset_index(drop=True)

    n = len(df)
    n_test = max(1, int(n * test_frac))
    n_train = n - n_test
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]

    X_train = train[HIST_FEATURE_COLS].astype(float)
    y_train = train[HIST_LABEL_COL].astype(float)
    X_test = test[HIST_FEATURE_COLS].astype(float)
    y_test = test[HIST_LABEL_COL].astype(float)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": 2 ** max_depth - 1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": seed,
    }
    model = lgb.train(
        params, train_ds,
        num_boost_round=n_estimators,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                   lgb.log_evaluation(0)],
    )

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    importance = dict(sorted(zip(
        HIST_FEATURE_COLS,
        model.feature_importance(importance_type="gain").tolist(),
    ), key=lambda kv: -kv[1]))

    # baseline = always-zero
    baseline_pred = np.zeros_like(y_test.values)
    baseline_rmse = _rmse(y_test, baseline_pred)

    test_predictions = pd.DataFrame({
        "t_news": test["t_news"].values,
        "ticker": test["ticker"].values,
        "event": test["event"].values,
        "y_true": y_test.values,
        "y_pred": pred_test,
    })

    return HistoricalTrainResult(
        model=model,
        feature_cols=list(HIST_FEATURE_COLS),
        train_size=len(train), test_size=len(test),
        train_rmse=_rmse(y_train, pred_train),
        test_rmse=_rmse(y_test, pred_test),
        train_mae=_mae(y_train, pred_train),
        test_mae=_mae(y_test, pred_test),
        train_r2=_r2(y_train, pred_train),
        test_r2=_r2(y_test, pred_test),
        baseline_test_rmse=baseline_rmse,
        feature_importance=importance,
        test_predictions=test_predictions,
    )
