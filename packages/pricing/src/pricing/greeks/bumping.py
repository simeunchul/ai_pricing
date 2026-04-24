"""Bumping (finite-difference) Greeks. Sanity check against analytic."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from pricing.bsm import BSMInputs, call_price, put_price
from pricing.greeks.analytic import Greeks


def bumping_greeks(
    i: BSMInputs,
    opt: str = "call",
    dS: float = 0.01,     # relative
    dSigma: float = 1e-4,
    dT: float = 1 / 365,
    dR: float = 1e-4,
) -> Greeks:
    pricer: Callable[[BSMInputs], float] = call_price if opt == "call" else put_price

    p0 = pricer(i)
    p_up = pricer(replace(i, S=i.S * (1 + dS)))
    p_dn = pricer(replace(i, S=i.S * (1 - dS)))

    delta = (p_up - p_dn) / (2 * i.S * dS)
    gamma = (p_up - 2 * p0 + p_dn) / (i.S * dS) ** 2
    vega = (pricer(replace(i, sigma=i.sigma + dSigma)) - p0) / dSigma
    theta = (pricer(replace(i, T=max(i.T - dT, 1e-9))) - p0) / (-dT)
    rho = (pricer(replace(i, r=i.r + dR)) - p0) / dR
    return Greeks(delta, gamma, vega, theta, rho)


__all__ = ["bumping_greeks"]
