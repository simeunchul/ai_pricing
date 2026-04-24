"""Monte Carlo pricing under GBM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricing.bsm import BSMInputs
from pricing.payoffs import PayoffFn


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    paths_used: int

    def ci95(self) -> tuple[float, float]:
        return (self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr)


def simulate_gbm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Return array of shape (n_paths, n_steps+1) including S0 at column 0."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal(size=(half, n_steps))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
        if n_paths % 2 == 1:
            Z = np.concatenate([Z, rng.standard_normal(size=(1, n_steps))], axis=0)
    else:
        Z = rng.standard_normal(size=(n_paths, n_steps))

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    log_increments = drift + diffusion * Z
    log_S = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_S)
    return paths


def mc_price(
    i: BSMInputs,
    payoff: PayoffFn,
    n_paths: int = 100_000,
    n_steps: int = 252,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    paths = simulate_gbm_paths(
        i.S, i.r, i.q, i.sigma, i.T, n_paths, n_steps, antithetic=antithetic, seed=seed
    )
    disc = math.exp(-i.r * i.T)
    payoffs = disc * payoff(paths)
    price = float(payoffs.mean())
    stderr = float(payoffs.std(ddof=1) / math.sqrt(n_paths))
    return MCResult(price, stderr, n_paths)


__all__ = ["MCResult", "simulate_gbm_paths", "mc_price"]
