"""Phase A2 — fair-value model: features → next-bucket return.

LightGBM regression. 입력: ETL.build_features() 의 출력 dataframe.
출력: 학습된 모델 + 검증 metrics (in-sample + simple holdout).

Walk-forward 정식 검증은 데이터가 일주일 이상 누적된 후 별도 모듈에서.
지금은 prototype: 시간 기준 80/20 split (앞 80% 학습, 뒤 20% holdout).

학습 시그너처:
  predicted_return_next = f(dev_bps_mean, dev_bps_std, momentum_bucket,
                            n_ticks, news_count_total, sentiment_weighted,
                            sentiment_max_abs_weighted, covered_weight,
                            symbol_id, hour_of_day, minute_of_hour)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "dev_bps_mean", "dev_bps_last", "dev_bps_std",
    "momentum_bucket", "n_ticks",
    "news_count_total", "sentiment_weighted", "sentiment_max_abs_weighted",
    "covered_weight",
    "hour", "minute",
    "symbol_id",
]
LABEL_COL = "label_return_next"


@dataclass
class TrainResult:
    model: Any                      # lightgbm.Booster
    feature_cols: list[str]
    train_size: int
    test_size: int
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    feature_importance: dict[str, float]
    test_predictions: pd.DataFrame  # ts_bucket, symbol, y_true, y_pred


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = pd.to_datetime(df["ts_bucket"]).dt.hour.astype(float)
    df["minute"] = pd.to_datetime(df["ts_bucket"]).dt.minute.astype(float)
    df["symbol_id"] = pd.Categorical(df["symbol"]).codes.astype(float)
    return df


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    if ss_tot <= 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def train_lightgbm(features_df: pd.DataFrame,
                    test_frac: float = 0.2,
                    n_estimators: int = 200,
                    learning_rate: float = 0.05,
                    max_depth: int = 5,
                    seed: int = 42) -> TrainResult:
    """Time-ordered train/test split + LightGBM regression."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError(
            "lightgbm 미설치. `pip install lightgbm` 후 재실행"
        ) from e

    df = _add_engineered_features(features_df)
    df = df.sort_values("ts_bucket").reset_index(drop=True)

    n = len(df)
    n_test = max(1, int(n * test_frac))
    n_train = n - n_test
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]

    X_train = train[FEATURE_COLS].astype(float)
    y_train = train[LABEL_COL].astype(float)
    X_test = test[FEATURE_COLS].astype(float)
    y_test = test[LABEL_COL].astype(float)

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_test, label=y_test, reference=train_ds)
    # 데이터가 작아서 (수백 rows) min_data_in_leaf 를 낮추고
    # early stopping 끄고 강제 round 수 학습 — 이렇게 안 하면
    # 첫 round 에서 즉시 멈추고 leaf=root 평균만 예측하게 됨.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": 2 ** max_depth - 1,
        "min_data_in_leaf": 5,
        "min_gain_to_split": 0.0,
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
        callbacks=[lgb.log_evaluation(0)],
    )

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    importance = dict(zip(
        FEATURE_COLS,
        model.feature_importance(importance_type="gain").tolist(),
    ))
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    test_predictions = pd.DataFrame({
        "ts_bucket": test["ts_bucket"].values,
        "symbol": test["symbol"].values,
        "y_true": y_test.values,
        "y_pred": pred_test,
    })

    return TrainResult(
        model=model,
        feature_cols=list(FEATURE_COLS),
        train_size=len(train),
        test_size=len(test),
        train_rmse=_rmse(y_train, pred_train),
        test_rmse=_rmse(y_test, pred_test),
        train_mae=_mae(y_train, pred_train),
        test_mae=_mae(y_test, pred_test),
        train_r2=_r2(y_train, pred_train),
        test_r2=_r2(y_test, pred_test),
        feature_importance=importance,
        test_predictions=test_predictions,
    )


def baseline_naive(features_df: pd.DataFrame, test_frac: float = 0.2) -> dict:
    """Baseline 비교: 항상 0 예측 (= 시장이 efficient 하다는 가정).

    XGBoost 가 baseline 보다 RMSE 낮으면 정보적 가치 있음.
    """
    df = features_df.sort_values("ts_bucket").reset_index(drop=True)
    n = len(df)
    n_test = max(1, int(n * test_frac))
    test = df.iloc[n - n_test:]
    y = test[LABEL_COL].astype(float).values
    pred = np.zeros_like(y)
    return {
        "test_rmse": _rmse(y, pred),
        "test_mae": _mae(y, pred),
        "test_r2": _r2(y, pred),
    }
