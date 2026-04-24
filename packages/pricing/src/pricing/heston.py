"""Heston stochastic volatility model.

Semi-analytic via Lewis-Lipton style Fourier integration for vanilla calls,
plus MC (full-truncation Euler) for arbitrary payoffs.

Layer B2 (Deep Calibration) uses this as ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from pricing.bsm import BSMInputs
from pricing.mc.engine import MCResult
from pricing.payoffs import PayoffFn


@dataclass(frozen=True)
class HestonParams:
    kappa: float   # mean reversion speed
    theta: float   # long-run variance
    xi: float      # vol-of-vol
    rho: float     # correlation between dW_S and dW_v
    v0: float      # initial variance


# ---------------------------------------------------------------------------
# Semi-analytic Heston call via "P1, P2" formulation (Heston 1993)
# Uses the "little Heston trap" form to avoid branch-cut issues (Albrecher 2007).
# ---------------------------------------------------------------------------


def _heston_char_fn(phi: complex, j: int, S: float, K: float, T: float, r: float,
                    p: HestonParams) -> complex:
    x = math.log(S)
    a = p.kappa * p.theta
    u = 0.5 if j == 1 else -0.5
    b = p.kappa - p.rho * p.xi if j == 1 else p.kappa

    d = np.sqrt((p.rho * p.xi * 1j * phi - b) ** 2 - p.xi**2 * (2 * u * 1j * phi - phi**2))
    g = (b - p.rho * p.xi * 1j * phi - d) / (b - p.rho * p.xi * 1j * phi + d)

    # Little Heston trap variant
    exp_dT = np.exp(-d * T)
    C = (r * 1j * phi * T
         + (a / p.xi**2) * ((b - p.rho * p.xi * 1j * phi - d) * T
                             - 2 * np.log((1 - g * exp_dT) / (1 - g))))
    D = ((b - p.rho * p.xi * 1j * phi - d) / p.xi**2) * ((1 - exp_dT) / (1 - g * exp_dT))
    return np.exp(C + D * p.v0 + 1j * phi * x)


def _heston_prob(j: int, S: float, K: float, T: float, r: float, p: HestonParams) -> float:
    def integrand(phi: float) -> float:
        num = np.exp(-1j * phi * math.log(K)) * _heston_char_fn(phi, j, S, K, T, r, p)
        return float((num / (1j * phi)).real)

    val, _ = quad(integrand, 1e-8, 100.0, limit=200)
    return 0.5 + val / math.pi


def heston_call_semi(S: float, K: float, T: float, r: float, p: HestonParams) -> float:
    """Semi-analytic Heston European call (no dividend)."""
    if T == 0:
        return max(S - K, 0.0)
    P1 = _heston_prob(1, S, K, T, r, p)
    P2 = _heston_prob(2, S, K, T, r, p)
    return S * P1 - K * math.exp(-r * T) * P2


def heston_put_semi(S: float, K: float, T: float, r: float, p: HestonParams) -> float:
    """Via put-call parity."""
    call = heston_call_semi(S, K, T, r, p)
    return call - S + K * math.exp(-r * T)


# ---------------------------------------------------------------------------
# Heston MC — full-truncation Euler (Lord et al. 2010)
# ---------------------------------------------------------------------------


def simulate_heston_paths(
    S0: float, r: float, T: float, p: HestonParams,
    n_paths: int, n_steps: int, seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    S = np.full(n_paths, S0)
    v = np.full(n_paths, p.v0)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        dW_S = Z1
        dW_v = p.rho * Z1 + math.sqrt(1 - p.rho**2) * Z2

        v_plus = np.maximum(v, 0.0)
        S = S * np.exp((r - 0.5 * v_plus) * dt + np.sqrt(v_plus) * sqrt_dt * dW_S)
        v = v + p.kappa * (p.theta - v_plus) * dt + p.xi * np.sqrt(v_plus) * sqrt_dt * dW_v
        paths[:, t + 1] = S
    return paths


def heston_mc(
    i: BSMInputs,
    p: HestonParams,
    payoff: PayoffFn,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> MCResult:
    paths = simulate_heston_paths(i.S, i.r, i.T, p, n_paths, n_steps, seed=seed)
    disc = math.exp(-i.r * i.T)
    payoffs = disc * payoff(paths)
    price = float(payoffs.mean())
    stderr = float(payoffs.std(ddof=1) / math.sqrt(n_paths))
    return MCResult(price, stderr, n_paths)


__all__ = [
    "HestonParams",
    "heston_call_semi",
    "heston_put_semi",
    "simulate_heston_paths",
    "heston_mc",
]
