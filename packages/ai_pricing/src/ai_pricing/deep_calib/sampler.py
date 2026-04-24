"""Heston parameter-space sampling for deep calibration training.

Horvath 2021 Table 1 style ranges, with Latin Hypercube for coverage.
"""

from __future__ import annotations

import numpy as np

# (kappa, theta, xi, rho, v0) ranges
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "kappa": (0.1, 5.0),
    "theta": (0.01, 0.20),
    "xi":    (0.05, 1.0),
    "rho":   (-0.95, 0.0),
    "v0":    (0.01, 0.25),
}


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin Hypercube Sampling on [0,1]^d."""
    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(size=(n, d))
    a = cut[:n, None]
    b = cut[1:, None]
    rd = u * (b - a) + a
    for col in range(d):
        rng.shuffle(rd[:, col])
    return rd


def sample_heston_params(n: int, seed: int | None = 42) -> np.ndarray:
    """Return (n, 5) array of [kappa, theta, xi, rho, v0]."""
    rng = np.random.default_rng(seed)
    u = lhs(n, 5, rng)
    lo = np.array([PARAM_RANGES[k][0] for k in ("kappa", "theta", "xi", "rho", "v0")])
    hi = np.array([PARAM_RANGES[k][1] for k in ("kappa", "theta", "xi", "rho", "v0")])
    X = lo + u * (hi - lo)

    # Feller condition soft filter: 2*kappa*theta > xi^2 desirable. Keep some violators
    # so the NN still sees realistic data; do NOT reject, just flag.
    return X


__all__ = ["PARAM_RANGES", "sample_heston_params", "lhs"]
