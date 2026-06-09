"""Minute-bar OHLCV fetcher with on-disk cache.

Backtest 용 historical 분봉 데이터 — yfinance 5m bars (~60 days history).
KIS vts 환경의 historical minute chart endpoint 가 500 을 반환해서 yfinance
fallback. 한국 ETF/대형주는 yfinance 의 .KS suffix 로 호출.

캐시 파일 형식: parquet, key = ticker + interval.
재실행 시 같은 ticker 의 갱신 데이터가 cache 와 partial overlap 이면 union.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[5] / "data" / "minute_bars"


def _to_yf_ticker(symbol: str) -> str:
    """KOSPI/KOSDAQ 6자리 → yfinance suffix.

    KOSDAQ 종목 도 .KS 로 가능한 경우 있고 (yfinance 가 둘 다 시도) .KQ
    suffix 가 정확. 일단 .KS 시도 후 .KQ fallback.
    """
    if symbol.isdigit() and len(symbol) == 6:
        return f"{symbol}.KS"
    return symbol


def _try_fetch(ticker_yf: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    t = yf.Ticker(ticker_yf)
    df = t.history(period=period, interval=interval, auto_adjust=False)
    return df


@dataclass
class MinuteBarsCache:
    """ticker 별 분봉 데이터 캐시. read-through, write-through."""
    cache_dir: Path = DEFAULT_CACHE_DIR
    interval: str = "5m"   # "1m" | "5m" | "15m"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}_{self.interval}.parquet"

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        p = self._path(symbol)
        df.to_parquet(p)


def fetch_minute_bars(
    symbol: str,
    period: str = "60d",
    interval: str = "5m",
    use_cache: bool = True,
    cache: MinuteBarsCache | None = None,
) -> pd.DataFrame:
    """Symbol 의 분봉 데이터 반환 (UTC+9 KST).

    cache hit 시 일단 cache 반환 (refresh 하려면 use_cache=False).
    cache miss 시 yfinance 호출 + cache 저장.

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.
        Index: tz-aware DatetimeIndex (Asia/Seoul).
    """
    cache = cache or MinuteBarsCache(interval=interval)
    if use_cache:
        cached = cache.load(symbol)
        if cached is not None and not cached.empty:
            return cached

    ticker_yf = _to_yf_ticker(symbol)
    df = _try_fetch(ticker_yf, period=period, interval=interval)

    # KOSDAQ 종목이 .KS 로 안 잡히면 .KQ 시도
    if df.empty and ticker_yf.endswith(".KS"):
        ticker_yf2 = ticker_yf.replace(".KS", ".KQ")
        df = _try_fetch(ticker_yf2, period=period, interval=interval)

    if df.empty:
        return df

    # KST timezone normalize (yfinance returns +09:00 already for KS suffix)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Seoul")
    else:
        df.index = df.index.tz_convert("Asia/Seoul")

    cache.save(symbol, df)
    return df


def get_bar_at_or_after(
    df: pd.DataFrame, target: datetime,
) -> pd.Series | None:
    """DataFrame 에서 target 시각 이후 첫 bar 반환. 없으면 None."""
    if df.empty:
        return None
    target_kst = pd.Timestamp(target).tz_convert("Asia/Seoul") \
        if pd.Timestamp(target).tzinfo else pd.Timestamp(target).tz_localize("Asia/Seoul")
    after = df.loc[df.index >= target_kst]
    if after.empty:
        return None
    return after.iloc[0]


__all__ = [
    "fetch_minute_bars",
    "MinuteBarsCache",
    "get_bar_at_or_after",
    "DEFAULT_CACHE_DIR",
]
