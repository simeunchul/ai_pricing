"""Cox-Ross-Rubinstein binomial tree. Handles European & American, call & put."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from pricing.bsm import BSMInputs


OptionType = Literal["call", "put"]


def _crr_params(sigma: float, dt: float, r: float, q: float) -> tuple[float, float, float]:
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    qp = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= qp <= 1.0):
        raise ValueError(f"Risk-neutral prob out of range: {qp:.4f}. Reduce N or check inputs.")
    return u, d, qp


def _payoff(S: np.ndarray, K: float, opt: OptionType) -> np.ndarray:
    return np.maximum(S - K, 0.0) if opt == "call" else np.maximum(K - S, 0.0)


def binomial_european(i: BSMInputs, N: int = 500, opt: OptionType = "call") -> float:
    if i.T == 0:
        return float(_payoff(np.array([i.S]), i.K, opt)[0])
    dt = i.T / N
    u, d, qp = _crr_params(i.sigma, dt, i.r, i.q)
    disc = math.exp(-i.r * dt)

    j = np.arange(N + 1)
    S_T = i.S * (u ** (N - j)) * (d**j)
    V = _payoff(S_T, i.K, opt)

    for _ in range(N):
        V = disc * (qp * V[:-1] + (1 - qp) * V[1:])
    return float(V[0])


def binomial_american(i: BSMInputs, N: int = 500, opt: OptionType = "put") -> float:
    if i.T == 0:
        return float(_payoff(np.array([i.S]), i.K, opt)[0])
    dt = i.T / N
    u, d, qp = _crr_params(i.sigma, dt, i.r, i.q)
    disc = math.exp(-i.r * dt)

    j = np.arange(N + 1)
    S = i.S * (u ** (N - j)) * (d**j)
    V = _payoff(S, i.K, opt)

    for step in range(N - 1, -1, -1):
        S = S[:-1] / u  # roll back one layer
        V = disc * (qp * V[:-1] + (1 - qp) * V[1:])
        intrinsic = _payoff(S, i.K, opt)
        V = np.maximum(V, intrinsic)
    return float(V[0])


__all__ = ["binomial_european", "binomial_american", "OptionType"]
