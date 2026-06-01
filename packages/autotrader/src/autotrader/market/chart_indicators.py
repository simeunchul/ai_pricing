"""차트 기술적 지표 — pandas 기반 (외부 lib 의존 최소화).

지원 지표:
  - RSI (Relative Strength Index, 14일 default)
  - MA (Moving Average, 5/20일)
  - MACD (12, 26, 9)
  - Bollinger Bands (20일, 2σ)
  - Returns (1/5/20일)
  - Volume MA + 거래량 비율
  - 변동성 (rolling std)

모든 함수는 pandas Series 입력, NaN 안전 처리.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss.

    Wilder's smoothing (EMA 변형).
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # loss = 0 (단조 상승) → RSI = 100
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    # gain = 0 (단조 하락) → RSI = 0
    out = out.where(~((avg_gain == 0) & (avg_loss > 0)), 0)
    return out


def moving_average(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=window).mean()


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    """Returns (upper, middle, lower)."""
    ma = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def bb_position(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """볼린저 밴드 내 위치 정규화 ([-1, +1]).

    -1 = 하단선, 0 = 중간선, +1 = 상단선.
    """
    upper, mid, lower = bollinger_bands(prices, window, num_std)
    band_width = (upper - lower) / 2
    return (prices - mid) / band_width.replace(0, np.nan)


def returns(prices: pd.Series, period: int) -> pd.Series:
    """N일 수익률."""
    return prices.pct_change(periods=period)


def volume_ratio(volumes: pd.Series, window: int = 20) -> pd.Series:
    """현재 거래량 / N일 평균 거래량."""
    vol_ma = volumes.rolling(window=window, min_periods=window).mean()
    return volumes / vol_ma.replace(0, np.nan)


def rolling_std(prices: pd.Series, window: int = 5) -> pd.Series:
    """N일 일별 수익률 표준편차 (변동성)."""
    daily_ret = prices.pct_change()
    return daily_ret.rolling(window=window, min_periods=window).std()


def add_chart_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + 외국인 데이터 DataFrame 에 모든 차트 지표 컬럼 추가.

    Input columns required:
      open, high, low, close, volume

    Output columns added:
      rsi_14, ma5, ma20, ma_ratio, macd_line, macd_signal, macd_hist,
      bb_upper, bb_lower, bb_pos,
      ret_1d, ret_5d, ret_20d,
      vol_ratio_20, std_5d
    """
    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    out["rsi_14"] = rsi(close, 14)
    out["ma5"] = moving_average(close, 5)
    out["ma20"] = moving_average(close, 20)
    out["ma_ratio"] = (out["ma5"] / out["ma20"]) - 1   # MA5/MA20 - 1

    macd_l, macd_s, macd_h = macd(close)
    out["macd_line"] = macd_l
    out["macd_signal"] = macd_s
    out["macd_hist"] = macd_h

    bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2.0)
    out["bb_upper"] = bb_u
    out["bb_lower"] = bb_l
    out["bb_pos"] = bb_position(close, 20, 2.0)

    out["ret_1d"] = returns(close, 1)
    out["ret_5d"] = returns(close, 5)
    out["ret_20d"] = returns(close, 20)

    out["vol_ratio_20"] = volume_ratio(volume, 20)
    out["std_5d"] = rolling_std(close, 5)

    return out


__all__ = [
    "rsi", "moving_average", "macd", "bollinger_bands", "bb_position",
    "returns", "volume_ratio", "rolling_std", "add_chart_features",
]
