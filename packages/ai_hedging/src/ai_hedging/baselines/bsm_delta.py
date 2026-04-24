"""Classical BSM Δ-hedge baseline. Used for sanity check (TC=0 → Deep Hedging
should converge to this) and as reference for TC>0 comparisons."""

from __future__ import annotations

import math

import numpy as np

from pricing.bsm import BSMInputs
from pricing.greeks.analytic import call_greeks, put_greeks

from ai_hedging.env import HedgingEnv, HedgingEnvConfig


class BSMDeltaHedger:
    def __init__(self, opt: str = "call"):
        self.opt = opt

    def act(self, env: HedgingEnv) -> float:
        dt_rem = env.cfg.T * (1 - env.step_idx / env.cfg.n_steps)
        T_eff = max(dt_rem, 1e-6)
        i = BSMInputs(env.S, env.cfg.K, T_eff, env.cfg.r, env.cfg.q, env.cfg.sigma)
        g = call_greeks(i) if self.opt == "call" else put_greeks(i)
        return float(g.delta)


def rebalance_interval(env_cfg: HedgingEnvConfig, every: int, hedger: BSMDeltaHedger,
                       n_paths: int = 2000, seed: int = 0) -> dict:
    """Run BSM Δ-hedger, rebalancing only every `every` steps. Returns terminal PnL stats."""
    env = HedgingEnv(env_cfg)
    terminals = []
    for p in range(n_paths):
        obs, _ = env.reset(seed=seed + p)
        hedge = 0.0
        done = False
        while not done:
            if env.step_idx % every == 0:
                hedge = hedger.act(env)
            obs, reward, terminated, truncated, info = env.step(np.array([hedge], dtype=np.float32))
            done = terminated or truncated
            if terminated:
                terminals.append(info["terminal_pnl"])
    arr = np.array(terminals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "pnl_array": arr,
    }


__all__ = ["BSMDeltaHedger", "rebalance_interval"]
