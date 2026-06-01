"""LightGBM 기반 외국인+기관 + 차트 지표 통합 신호 분류기.

목표: 어제 features (외국인 ratio, 기관 ratio, 차트 지표) 로 다음날 (T+1) 가격이
      +0.5% 이상 상승할 확률 예측.

학습 데이터: 40 종목 × 1200 거래일 = 약 48,000 sample.
Train/test split: walk-forward (80% train, 20% test, expanding window 도 가능).

Features (15):
  flow_ratio, inst_ratio, flow_ma5, inst_ma5, flow_inst_product,
  rsi_14, ma_ratio, macd_line, macd_signal, bb_pos,
  ret_1d, ret_5d, ret_20d, vol_ratio_20, std_5d

Target: T+1 수익률 ≥ +0.5% (binary classification)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "flow_ratio", "inst_ratio", "flow_ma5", "inst_ma5", "flow_inst_product",
    "rsi_14", "ma_ratio", "macd_line", "macd_signal", "bb_pos",
    "ret_1d", "ret_5d", "ret_20d", "vol_ratio_20", "std_5d",
]


def _add_flow_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """flow_ratio / inst_ratio 기반 ML feature 추가 (5일 MA + interaction)."""
    out = df.copy()
    out["flow_ma5"] = out["flow_ratio"].rolling(window=5, min_periods=5).mean()
    out["inst_ma5"] = out["inst_ratio"].rolling(window=5, min_periods=5).mean()
    out["flow_inst_product"] = out["flow_ratio"] * out["inst_ratio"]
    return out


def build_features(symbol_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """모든 종목의 일별 데이터 → ML 학습용 long-format DataFrame.

    Returns DataFrame with: symbol, date, FEATURE_COLS, future_return, target
      target = 1 if future_return >= 0.005 else 0
    """
    from autotrader.market.chart_indicators import add_chart_features

    rows = []
    for sym, df in symbol_data.items():
        if df.empty:
            continue
        with_chart = add_chart_features(df)
        with_features = _add_flow_ml_features(with_chart)
        # T+1 수익률 (target)
        with_features["future_return"] = with_features["close"].pct_change().shift(-1)
        with_features["target"] = (with_features["future_return"] >= 0.005).astype(int)

        # 필요한 컬럼만 추출, NaN 행 drop
        keep_cols = ["symbol", "date"] + FEATURE_COLS + ["future_return", "target"]
        sub = with_features.copy()
        sub["symbol"] = sym
        sub["date"] = sub.index
        sub = sub[keep_cols].reset_index(drop=True)
        sub = sub.dropna(subset=FEATURE_COLS + ["future_return"])
        rows.append(sub)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


@dataclass
class TrainResult:
    model: object                    # lightgbm.Booster
    feature_importance: dict[str, float]
    train_auc: float
    test_auc: float
    test_accuracy: float
    test_precision: float           # 양성 예측의 정확도
    n_train: int
    n_test: int
    test_predictions: pd.DataFrame  # date, symbol, pred_proba, target


def train_lightgbm(
    features_df: pd.DataFrame,
    train_split_date: str | None = None,
    train_frac: float = 0.8,
    params: dict | None = None,
) -> TrainResult:
    """LightGBM binary classifier 학습.

    Args:
        features_df: build_features 결과
        train_split_date: 명시 시 그 날짜 미만 = train, 이상 = test
        train_frac: split_date None 일 때 시간순 비율
        params: LightGBM hyperparams
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

    df = features_df.sort_values("date").reset_index(drop=True)

    # train/test split (시간순)
    if train_split_date:
        cutoff = pd.Timestamp(train_split_date)
        train = df[df["date"] < cutoff]
        test = df[df["date"] >= cutoff]
    else:
        cutoff_idx = int(len(df) * train_frac)
        train = df.iloc[:cutoff_idx]
        test = df.iloc[cutoff_idx:]

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values

    default_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    valid_data = lgb.Dataset(X_test, label=y_test, feature_name=FEATURE_COLS,
                              reference=train_data)

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),     # silent
        ],
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_auc = float(roc_auc_score(y_train, train_pred)) if len(set(y_train)) > 1 else float("nan")
    test_auc = float(roc_auc_score(y_test, test_pred)) if len(set(y_test)) > 1 else float("nan")
    test_label = (test_pred >= 0.5).astype(int)
    test_acc = float(accuracy_score(y_test, test_label))
    test_prec = float(precision_score(y_test, test_label, zero_division=0))

    importance = dict(zip(
        FEATURE_COLS,
        model.feature_importance(importance_type="gain").tolist(),
    ))

    test_pred_df = test[["date", "symbol", "target"]].copy()
    test_pred_df["pred_proba"] = test_pred

    return TrainResult(
        model=model,
        feature_importance=importance,
        train_auc=train_auc,
        test_auc=test_auc,
        test_accuracy=test_acc,
        test_precision=test_prec,
        n_train=len(train),
        n_test=len(test),
        test_predictions=test_pred_df.reset_index(drop=True),
    )


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_score(model, features: pd.DataFrame) -> np.ndarray:
    """학습된 모델로 점수 예측. features 는 FEATURE_COLS 컬럼 가져야."""
    X = features[FEATURE_COLS].values
    return model.predict(X)


__all__ = [
    "FEATURE_COLS", "build_features", "TrainResult",
    "train_lightgbm", "save_model", "load_model", "predict_score",
]
