"""Common metrics for pricing and hedging comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PricingStats:
    method: str
    mean_err: float
    max_err: float
    mean_rel_err: float
    p95_rel_err: float
    inference_ms: float

    def as_row(self) -> dict:
        return self.__dict__


@dataclass(frozen=True)
class HedgingStats:
    method: str
    mean_pnl: float
    std_pnl: float
    cvar_5: float        # positive = tail loss magnitude
    sharpe: float
    max_dd: float
    turnover: float
    inference_ms: float

    def as_row(self) -> dict:
        return self.__dict__


def pricing_stats_from_arrays(
    method: str,
    preds: np.ndarray,
    truth: np.ndarray,
    inference_ms: float,
) -> PricingStats:
    err = preds - truth
    rel = np.abs(err) / np.maximum(np.abs(truth), 1e-6)
    return PricingStats(
        method=method,
        mean_err=float(np.abs(err).mean()),
        max_err=float(np.abs(err).max()),
        mean_rel_err=float(rel.mean()),
        p95_rel_err=float(np.quantile(rel, 0.95)),
        inference_ms=inference_ms,
    )


def hedging_stats_from_arrays(
    method: str,
    pnl: np.ndarray,
    turnover: float,
    inference_ms: float,
    cvar_alpha: float = 0.05,
) -> HedgingStats:
    q = np.quantile(pnl, cvar_alpha)
    tail = pnl[pnl <= q]
    cvar = float(-tail.mean()) if len(tail) else float(-q)
    sr = float(pnl.mean() / pnl.std()) if pnl.std() > 1e-12 else 0.0

    # Drawdown based on cumulative sum
    eq = np.cumsum(pnl)
    cummax = np.maximum.accumulate(eq)
    dd = (eq - cummax).min()

    return HedgingStats(
        method=method,
        mean_pnl=float(pnl.mean()),
        std_pnl=float(pnl.std()),
        cvar_5=cvar,
        sharpe=sr,
        max_dd=float(dd),
        turnover=turnover,
        inference_ms=inference_ms,
    )


__all__ = [
    "PricingStats",
    "HedgingStats",
    "pricing_stats_from_arrays",
    "hedging_stats_from_arrays",
]
