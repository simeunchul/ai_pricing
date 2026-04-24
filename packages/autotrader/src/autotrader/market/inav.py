"""ETF iNAV estimator from constituent basket prices.

The true iNAV is published by the issuer; we reconstruct it from underlying
constituents + weights to compute a realtime ETF vs iNAV deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InavEstimator:
    """Weights should sum to ~1.0; any residual is treated as cash.

    Example (KODEX 200 partial proxy with top 5 names):
        weights = {"005930": 0.30, "000660": 0.10, "207940": 0.05, ...}
    """
    constituents: dict[str, float]
    cash_weight: float = 0.0           # remaining weight in cash
    issuer_factor: float = 1.0         # scale published iNAV (prov)

    def estimate(self, prices: dict[str, float]) -> float:
        total = 0.0
        for sym, w in self.constituents.items():
            p = prices.get(sym)
            if p is None:
                raise KeyError(f"Missing constituent price: {sym}")
            total += w * p
        total += self.cash_weight
        return total * self.issuer_factor


def inav_from_basket(
    weights: dict[str, float],
    prices: dict[str, float],
    cash_weight: float = 0.0,
) -> float:
    return InavEstimator(constituents=weights, cash_weight=cash_weight).estimate(prices)


def deviation(etf_price: float, inav: float) -> float:
    """Signed percentage deviation. Positive = ETF trades above fair value."""
    if inav <= 0:
        return 0.0
    return (etf_price - inav) / inav


__all__ = ["InavEstimator", "inav_from_basket", "deviation"]
