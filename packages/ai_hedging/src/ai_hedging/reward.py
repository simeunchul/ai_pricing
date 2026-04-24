"""Risk-aware losses used as training / evaluation objectives."""

from __future__ import annotations

import numpy as np


def cvar_loss(pnl: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional Value at Risk at level alpha (tail-expected loss).

    Positive number = size of expected loss in worst alpha fraction.
    """
    pnl = np.asarray(pnl)
    q = np.quantile(pnl, alpha)
    tail = pnl[pnl <= q]
    if len(tail) == 0:
        return float(-q)
    return float(-tail.mean())


def mean_var_loss(pnl: np.ndarray, lam: float = 1.0) -> float:
    """E[-PnL] + lam * Var[PnL]."""
    pnl = np.asarray(pnl)
    return float(-pnl.mean() + lam * pnl.var())


def sharpe(pnl: np.ndarray, ann_factor: float = 252.0) -> float:
    pnl = np.asarray(pnl)
    if pnl.std() < 1e-12:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(ann_factor))


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity)
    cum_max = np.maximum.accumulate(equity)
    dd = (equity - cum_max) / np.where(cum_max != 0, cum_max, 1)
    return float(dd.min())


__all__ = ["cvar_loss", "mean_var_loss", "sharpe", "max_drawdown"]
