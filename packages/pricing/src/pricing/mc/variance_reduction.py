"""Variance reduction helpers."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from pricing.bsm import BSMInputs, call_price
from pricing.mc.engine import MCResult, simulate_gbm_paths
from pricing.payoffs import PayoffFn


def antithetic_normals(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Generate Z and concatenate with -Z along first axis."""
    Z = rng.standard_normal(size=shape)
    return np.concatenate([Z, -Z], axis=0)


def control_variate_call(
    i: BSMInputs,
    payoff: PayoffFn,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> MCResult:
    """Use European call terminal payoff as control variate (BSM closed-form known)."""
    paths = simulate_gbm_paths(
        i.S, i.r, i.q, i.sigma, i.T, n_paths, n_steps, antithetic=True, seed=seed
    )
    disc = math.exp(-i.r * i.T)

    target = disc * payoff(paths)
    control_terminal = np.maximum(paths[:, -1] - i.K, 0.0) * disc
    control_expected = call_price(i)

    cov = np.cov(target, control_terminal, ddof=1)
    b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    adjusted = target - b * (control_terminal - control_expected)

    price = float(adjusted.mean())
    stderr = float(adjusted.std(ddof=1) / math.sqrt(n_paths))
    return MCResult(price, stderr, n_paths)


__all__ = ["antithetic_normals", "control_variate_call"]
