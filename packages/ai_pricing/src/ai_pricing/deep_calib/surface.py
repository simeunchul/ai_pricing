"""Generate an IV surface from Heston params via semi-analytic pricing + IV solver."""

from __future__ import annotations

import numpy as np

from pricing.bsm import BSMInputs
from pricing.heston import HestonParams, heston_call_semi
from pricing.iv import implied_vol

# Grid: 5 moneyness × 5 maturities = 25 points
STRIKES = np.array([0.80, 0.90, 1.00, 1.10, 1.20])       # K/S
MATURITIES = np.array([1 / 12, 3 / 12, 6 / 12, 12 / 12, 24 / 12])

N_STRIKES = len(STRIKES)
N_MATS = len(MATURITIES)
N_POINTS = N_STRIKES * N_MATS


def iv_surface(p: HestonParams, S: float = 1.0, r: float = 0.02) -> np.ndarray:
    """Return flat vector of 25 implied vols (row-major: maturity × strike)."""
    out = np.empty(N_POINTS)
    idx = 0
    for T in MATURITIES:
        for k in STRIKES:
            K = k * S
            price = heston_call_semi(S, K, T, r, p)
            # intrinsic guard
            intrinsic = max(S - K * np.exp(-r * T), 0.0)
            price = max(price, intrinsic + 1e-8)
            try:
                iv = implied_vol(price, BSMInputs(S, K, T, r, 0.0, 0.2), opt="call")
            except ValueError:
                iv = np.nan
            out[idx] = iv
            idx += 1
    return out


def iv_surface_batch(params: np.ndarray, S: float = 1.0, r: float = 0.02) -> np.ndarray:
    """Vectorize by looping over rows — Heston char-fn is per-row, so parallelism happens per call."""
    n = len(params)
    out = np.empty((n, N_POINTS))
    for i in range(n):
        kappa, theta, xi, rho, v0 = params[i]
        p = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
        out[i] = iv_surface(p, S=S, r=r)
    return out


__all__ = ["STRIKES", "MATURITIES", "N_STRIKES", "N_MATS", "N_POINTS", "iv_surface", "iv_surface_batch"]
