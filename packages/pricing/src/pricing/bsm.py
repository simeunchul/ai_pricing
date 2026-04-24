"""Black-Scholes-Merton closed-form pricing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class BSMInputs:
    """Standard BSM inputs.

    S: spot, K: strike, T: time-to-maturity (years), r: risk-free, q: dividend yield,
    sigma: volatility.
    """

    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float

    def __post_init__(self) -> None:
        if self.T < 0:
            raise ValueError("T must be >= 0")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")


def _d1_d2(i: BSMInputs) -> tuple[float, float]:
    if i.T == 0 or i.sigma == 0:
        return float("inf") if i.S > i.K else float("-inf"), 0.0
    vol_t = i.sigma * math.sqrt(i.T)
    d1 = (math.log(i.S / i.K) + (i.r - i.q + 0.5 * i.sigma**2) * i.T) / vol_t
    d2 = d1 - vol_t
    return d1, d2


def call_price(i: BSMInputs) -> float:
    if i.T == 0:
        return max(i.S - i.K, 0.0)
    d1, d2 = _d1_d2(i)
    return i.S * math.exp(-i.q * i.T) * norm.cdf(d1) - i.K * math.exp(-i.r * i.T) * norm.cdf(d2)


def put_price(i: BSMInputs) -> float:
    if i.T == 0:
        return max(i.K - i.S, 0.0)
    d1, d2 = _d1_d2(i)
    return i.K * math.exp(-i.r * i.T) * norm.cdf(-d2) - i.S * math.exp(-i.q * i.T) * norm.cdf(-d1)


def forward_price(i: BSMInputs) -> float:
    return i.S * math.exp((i.r - i.q) * i.T)


__all__ = ["BSMInputs", "call_price", "put_price", "forward_price"]
