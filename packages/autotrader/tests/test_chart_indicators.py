"""차트 지표 단위 테스트."""

from __future__ import annotations

import pandas as pd
import numpy as np

from autotrader.market.chart_indicators import (
    rsi, moving_average, macd, bollinger_bands, bb_position,
    returns, volume_ratio, rolling_std, add_chart_features,
)


def test_rsi_constant_price_returns_nan_or_50():
    """가격이 일정하면 RSI 정의 모호 (보통 NaN 또는 50)."""
    prices = pd.Series([100.0] * 30)
    out = rsi(prices, 14)
    assert pd.isna(out.iloc[-1]) or 49 <= out.iloc[-1] <= 51


def test_rsi_strong_uptrend_high():
    """단조 상승 → RSI 80~100."""
    prices = pd.Series([100 + i for i in range(30)], dtype=float)
    out = rsi(prices, 14)
    assert out.iloc[-1] > 80


def test_rsi_strong_downtrend_low():
    """단조 하락 → RSI 0~20."""
    prices = pd.Series([100 - i * 0.5 for i in range(30)], dtype=float)
    out = rsi(prices, 14)
    assert out.iloc[-1] < 20


def test_moving_average_basic():
    prices = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ma5 = moving_average(prices, 5)
    # ma5 of last 5 = 6,7,8,9,10 = 8.0
    assert ma5.iloc[-1] == 8.0
    # 처음 4개는 NaN
    assert pd.isna(ma5.iloc[3])


def test_macd_returns_3_series():
    prices = pd.Series([100 + i for i in range(50)], dtype=float)
    line, signal, hist = macd(prices)
    assert len(line) == len(prices)
    assert len(signal) == len(prices)
    # 상승 trend → MACD line > 0
    assert line.iloc[-1] > 0


def test_bollinger_bands_envelope():
    prices = pd.Series([100 + np.sin(i / 5) * 10 for i in range(50)], dtype=float)
    upper, mid, lower = bollinger_bands(prices, 20, 2.0)
    # upper > mid > lower
    valid = ~upper.isna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_bb_position_normalized():
    prices = pd.Series([100 + np.sin(i / 5) * 10 for i in range(50)], dtype=float)
    pos = bb_position(prices, 20, 2.0)
    valid = ~pos.isna()
    # bb_pos 값은 보통 [-1.5, +1.5] 범위 (2σ 밴드 + 약간 over)
    assert pos[valid].min() > -3.0
    assert pos[valid].max() < 3.0


def test_returns_simple():
    prices = pd.Series([100, 105, 110, 115, 120], dtype=float)
    r1 = returns(prices, 1)
    # 100 → 105 = +5%
    assert abs(r1.iloc[1] - 0.05) < 1e-6


def test_volume_ratio():
    """마지막날 거래량 3000, 평균 (19×1000 + 3000)/20 = 1100, ratio = 2.727."""
    vol = pd.Series([1000] * 19 + [3000], dtype=float)
    ratio = volume_ratio(vol, 20)
    assert abs(ratio.iloc[-1] - 2.727) < 0.05


def test_add_chart_features_creates_all_columns():
    """add_chart_features 후 모든 지표 컬럼이 만들어지는지."""
    n = 60
    df = pd.DataFrame({
        "open": np.random.uniform(95, 105, n),
        "high": np.random.uniform(100, 110, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(95, 105, n),
        "volume": np.random.uniform(1e6, 1e7, n),
    })
    out = add_chart_features(df)
    expected = [
        "rsi_14", "ma5", "ma20", "ma_ratio",
        "macd_line", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_pos",
        "ret_1d", "ret_5d", "ret_20d",
        "vol_ratio_20", "std_5d",
    ]
    for col in expected:
        assert col in out.columns, f"missing {col}"
    # 마지막 행은 모든 지표 valid (lookback 충분)
    assert not out[expected].iloc[-1].isna().any()


def test_add_chart_features_short_data_handles_nan():
    """data 부족 시 NaN 들어가지만 crash 안 함."""
    df = pd.DataFrame({
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100, 101, 102],
        "volume": [1e6, 1.1e6, 1.2e6],
    })
    out = add_chart_features(df)
    # 짧은 데이터라 ma20 등은 모두 NaN
    assert out["ma20"].isna().all()
    # 그러나 컬럼 자체는 존재
    assert "ma20" in out.columns
