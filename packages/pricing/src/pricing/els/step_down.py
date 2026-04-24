"""Step-down Auto-call ELS (Equity-Linked Security).

Standard Korean retail structure:
  - 2 or 3 underlying assets
  - Every 6 months, check if min(S/S0) >= barrier[k] → early redemption with cumulative coupon
  - At maturity: if KI never hit, pay notional + full coupon; else pay min-performance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pricing.mc.engine import MCResult


@dataclass
class StepDownELS:
    S0: np.ndarray                  # shape (n_assets,)
    barriers: list[float]           # e.g. [0.90, 0.85, 0.85, 0.80, 0.80, 0.75] (6 semi-annual)
    ki_barrier: float               # e.g. 0.50 (KI when min(S/S0) <= this at ANY time)
    coupon_rate: float              # per-period (e.g. 0.06 per 6mo)
    maturity_years: float           # e.g. 3.0
    obs_per_year: int = 2           # 6-month obs = 2/yr
    notional: float = 10_000.0

    def obs_times(self) -> np.ndarray:
        n_obs = int(self.maturity_years * self.obs_per_year)
        return np.linspace(self.maturity_years / n_obs, self.maturity_years, n_obs)


@dataclass(frozen=True)
class ELSResult:
    price: float
    stderr: float
    autocall_prob: list[float]      # per-period
    ki_hit_prob: float
    expected_life: float            # years


def price_els(
    product: StepDownELS,
    r: float,
    q: np.ndarray,                  # dividend yield per asset
    sigma: np.ndarray,              # vol per asset
    corr: np.ndarray,               # correlation matrix (n_assets, n_assets)
    n_paths: int = 50_000,
    n_steps_per_year: int = 252,
    seed: int | None = None,
    ki_check: str = "daily",        # "daily" uses each step; "obs" only at obs dates
) -> ELSResult:
    rng = np.random.default_rng(seed)
    n_assets = len(product.S0)
    n_steps = int(product.maturity_years * n_steps_per_year)
    dt = product.maturity_years / n_steps
    obs_idx = np.round(product.obs_times() * n_steps_per_year).astype(int) - 1
    obs_idx = np.clip(obs_idx, 0, n_steps - 1)

    # Cholesky for correlated BM
    L = np.linalg.cholesky(corr)

    S = np.broadcast_to(product.S0, (n_paths, n_assets)).copy()
    min_ratio_run = np.ones(n_paths)  # track min(S/S0) across all observed times

    redeemed = np.zeros(n_paths, dtype=bool)
    redemption_time = np.full(n_paths, product.maturity_years)
    redemption_payoff = np.zeros(n_paths)
    autocall_counts = [0] * len(obs_idx)

    ki_hit = np.zeros(n_paths, dtype=bool)

    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * math.sqrt(dt)  # shape (n_assets,)

    for step in range(n_steps):
        Z = rng.standard_normal(size=(n_paths, n_assets))
        Zc = Z @ L.T
        S = S * np.exp(drift + diff * Zc)

        ratio = (S / product.S0).min(axis=1)
        if ki_check == "daily":
            ki_hit |= (ratio <= product.ki_barrier)
        min_ratio_run = np.minimum(min_ratio_run, ratio)

        if step in obs_idx:
            k = list(obs_idx).index(step)
            barrier_k = product.barriers[k]
            not_yet = ~redeemed
            hits = not_yet & (ratio >= barrier_k)
            n_hits = int(hits.sum())
            autocall_counts[k] = n_hits
            if n_hits > 0:
                t_k = product.obs_times()[k]
                coupon = (k + 1) * product.coupon_rate
                payoff = product.notional * (1.0 + coupon)
                redemption_time[hits] = t_k
                redemption_payoff[hits] = payoff
                redeemed |= hits

    # At maturity, for non-redeemed paths:
    final_ratio = (S / product.S0).min(axis=1)
    if ki_check == "obs":
        # final KI check: hit if any obs min_ratio <= ki_barrier
        ki_hit = (min_ratio_run <= product.ki_barrier)

    alive = ~redeemed
    for idx in np.where(alive)[0]:
        if ki_hit[idx]:
            # min-performance payoff
            redemption_payoff[idx] = product.notional * final_ratio[idx]
        else:
            n_obs = len(obs_idx)
            coupon = n_obs * product.coupon_rate
            redemption_payoff[idx] = product.notional * (1.0 + coupon)

    # Discount to t=0
    pv = redemption_payoff * np.exp(-r * redemption_time)
    price = float(pv.mean())
    stderr = float(pv.std(ddof=1) / math.sqrt(n_paths))
    autocall_prob = [c / n_paths for c in autocall_counts]
    ki_prob = float(ki_hit.mean())
    expected_life = float(redemption_time.mean())
    return ELSResult(price, stderr, autocall_prob, ki_prob, expected_life)


__all__ = ["StepDownELS", "ELSResult", "price_els"]
