"""News-driven fair-value signal layer for ETF arbitrage bot.

Phase A — fair-price L3 pricer:
  - ETL: ETF tick log + classified news → 5-min bucket features
  - Model: XGBoost regression → next-bucket return per stock
  - Aggregation: Σ wᵢ × predicted_pᵢ × factor → iNAV_fair
  - Hook: bot decision = combine basket dev + fair dev

This module is in pre-production (training data still accumulating). Hook into
production runner only after walk-forward validation passes.
"""
from autotrader.news_signal.etl import (
    ETF_UNDERLYING,
    ETL_Config,
    build_features,
    bucket_news,
    bucket_ticks,
)

__all__ = [
    "ETF_UNDERLYING",
    "ETL_Config",
    "build_features",
    "bucket_news",
    "bucket_ticks",
]
