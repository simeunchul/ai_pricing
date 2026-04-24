"""Closed-form Greeks for BSM."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from pricing.bsm import BSMInputs, _d1_d2


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float   # per 1.0 change in sigma (not per 1%)
    theta: float  # per year
    rho: float    # per 1.0 change in r


def call_greeks(i: BSMInputs) -> Greeks:
    if i.T == 0:
        return Greeks(1.0 if i.S > i.K else 0.0, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1_d2(i)
    sqrt_t = math.sqrt(i.T)
    disc_r = math.exp(-i.r * i.T)
    disc_q = math.exp(-i.q * i.T)
    pdf_d1 = norm.pdf(d1)

    delta = disc_q * norm.cdf(d1)
    gamma = disc_q * pdf_d1 / (i.S * i.sigma * sqrt_t)
    vega = i.S * disc_q * pdf_d1 * sqrt_t
    theta = (
        -i.S * disc_q * pdf_d1 * i.sigma / (2 * sqrt_t)
        - i.r * i.K * disc_r * norm.cdf(d2)
        + i.q * i.S * disc_q * norm.cdf(d1)
    )
    rho = i.K * i.T * disc_r * norm.cdf(d2)
    return Greeks(delta, gamma, vega, theta, rho)


def put_greeks(i: BSMInputs) -> Greeks:
    if i.T == 0:
        return Greeks(-1.0 if i.S < i.K else 0.0, 0.0, 0.0, 0.0, 0.0)
    d1, d2 = _d1_d2(i)
    sqrt_t = math.sqrt(i.T)
    disc_r = math.exp(-i.r * i.T)
    disc_q = math.exp(-i.q * i.T)
    pdf_d1 = norm.pdf(d1)

    delta = disc_q * (norm.cdf(d1) - 1.0)
    gamma = disc_q * pdf_d1 / (i.S * i.sigma * sqrt_t)
    vega = i.S * disc_q * pdf_d1 * sqrt_t
    theta = (
        -i.S * disc_q * pdf_d1 * i.sigma / (2 * sqrt_t)
        + i.r * i.K * disc_r * norm.cdf(-d2)
        - i.q * i.S * disc_q * norm.cdf(-d1)
    )
    rho = -i.K * i.T * disc_r * norm.cdf(-d2)
    return Greeks(delta, gamma, vega, theta, rho)


__all__ = ["Greeks", "call_greeks", "put_greeks"]
