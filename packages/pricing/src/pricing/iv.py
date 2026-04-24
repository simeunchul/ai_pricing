"""Implied volatility via Brent's method."""

from __future__ import annotations

from dataclasses import replace

from scipy.optimize import brentq

from pricing.bsm import BSMInputs, call_price, put_price


def implied_vol(
    target_price: float,
    i: BSMInputs,
    opt: str = "call",
    lo: float = 1e-6,
    hi: float = 5.0,
    tol: float = 1e-8,
) -> float:
    """Solve for sigma so that BSM price == target_price."""
    pricer = call_price if opt == "call" else put_price

    def f(sig: float) -> float:
        return pricer(replace(i, sigma=sig)) - target_price

    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        # widen
        for cap in (10.0, 20.0, 50.0):
            f_hi = f(cap)
            if f_lo * f_hi <= 0:
                hi = cap
                break
        else:
            raise ValueError(
                f"IV solver: no sign change in [{lo}, 50]. target={target_price}, inputs={i}"
            )
    return float(brentq(f, lo, hi, xtol=tol))


__all__ = ["implied_vol"]
