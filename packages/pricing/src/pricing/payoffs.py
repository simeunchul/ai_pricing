"""Payoff functions for derivatives. All take a 1D or 2D ndarray of terminal/path prices."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class PayoffFn(Protocol):
    def __call__(self, paths: np.ndarray) -> np.ndarray: ...


def call_payoff(K: float) -> PayoffFn:
    """European call: max(S_T - K, 0)."""

    def f(paths: np.ndarray) -> np.ndarray:
        S_T = paths[..., -1] if paths.ndim > 1 else paths
        return np.maximum(S_T - K, 0.0)

    return f


def put_payoff(K: float) -> PayoffFn:
    def f(paths: np.ndarray) -> np.ndarray:
        S_T = paths[..., -1] if paths.ndim > 1 else paths
        return np.maximum(K - S_T, 0.0)

    return f


def digital_call(K: float, Q: float = 1.0) -> PayoffFn:
    def f(paths: np.ndarray) -> np.ndarray:
        S_T = paths[..., -1] if paths.ndim > 1 else paths
        return (S_T > K).astype(float) * Q

    return f


def asian_arith_call(K: float) -> PayoffFn:
    """Arithmetic-average Asian call. Requires full path."""

    def f(paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 1:
            raise ValueError("Asian option needs full path (2D array).")
        avg = paths.mean(axis=-1)
        return np.maximum(avg - K, 0.0)

    return f


def barrier_up_and_out_call(K: float, B: float) -> PayoffFn:
    def f(paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 1:
            raise ValueError("Barrier option needs full path.")
        breached = (paths.max(axis=-1) >= B)
        return np.where(breached, 0.0, np.maximum(paths[..., -1] - K, 0.0))

    return f


__all__ = [
    "PayoffFn",
    "call_payoff",
    "put_payoff",
    "digital_call",
    "asian_arith_call",
    "barrier_up_and_out_call",
]
