"""Gym(nasium) environment for Deep Hedging.

We hedge a short European call under GBM with proportional transaction cost.
Episode: one option lifetime discretized into `n_steps`.

State (normalized, dim 5):
    [log(S/K), T_remaining/T_total, current_hedge, sigma*sqrt(T_rem), cash_norm]

Action: Box([-1, 2]) — desired hedge in shares per unit notional.
    (hedge > 1 allowed for gamma-scalping; negative for puts.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gym = None      # type: ignore
    spaces = None   # type: ignore

from pricing.bsm import BSMInputs, call_price


@dataclass
class HedgingEnvConfig:
    S0: float = 100.0
    K: float = 100.0
    T: float = 30 / 365
    r: float = 0.02
    q: float = 0.0
    sigma: float = 0.20
    n_steps: int = 30
    tc_rate: float = 0.0      # proportional transaction cost
    notional: float = 1.0
    opt: str = "call"
    seed: int | None = None


class HedgingEnv(gym.Env if _HAS_GYM else object):
    """Gym env. Short 1 call, choose hedge each step. Reward = -per-step PnL variance."""

    metadata = {"render_modes": []} if _HAS_GYM else {}

    def __init__(self, cfg: HedgingEnvConfig | None = None):
        if not _HAS_GYM:
            raise ImportError("pip install gymnasium stable-baselines3 torch")
        super().__init__()
        self.cfg = cfg or HedgingEnvConfig()
        self.action_space = spaces.Box(low=-1.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32,
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self._reset_state()

    def _reset_state(self):
        self.step_idx = 0
        self.S = self.cfg.S0
        self.hedge = 0.0
        self.cash = call_price(BSMInputs(
            self.cfg.S0, self.cfg.K, self.cfg.T, self.cfg.r, self.cfg.q, self.cfg.sigma))
        self.initial_premium = self.cash
        self.step_pnl = 0.0
        self.cum_tc = 0.0
        self.cum_hedge_turnover = 0.0
        self.done = False

    def _obs(self) -> np.ndarray:
        dt_rem = self.cfg.T * (1 - self.step_idx / self.cfg.n_steps)
        return np.array([
            math.log(max(self.S / self.cfg.K, 1e-9)),
            dt_rem / self.cfg.T,
            self.hedge,
            self.cfg.sigma * math.sqrt(max(dt_rem, 1e-6)),
            self.cash / max(self.initial_premium, 1e-6),
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        assert not self.done, "Call reset() first"
        action = float(np.clip(action, self.action_space.low, self.action_space.high)[0])

        # transaction cost on hedge change
        dH = action - self.hedge
        tc = self.cfg.tc_rate * abs(dH) * self.S
        self.cash -= tc
        self.cum_tc += tc
        self.cum_hedge_turnover += abs(dH)
        self.hedge = action

        # advance underlying one step (GBM)
        dt = self.cfg.T / self.cfg.n_steps
        Z = self.rng.standard_normal()
        S_next = self.S * math.exp(
            (self.cfg.r - self.cfg.q - 0.5 * self.cfg.sigma**2) * dt
            + self.cfg.sigma * math.sqrt(dt) * Z
        )

        # PnL accumulation: hedge gains, cash grows at r
        hedge_pnl = self.hedge * (S_next - self.S)
        self.cash = self.cash * math.exp(self.cfg.r * dt) + hedge_pnl
        self.S = S_next
        self.step_idx += 1

        terminated = self.step_idx >= self.cfg.n_steps
        truncated = False

        if terminated:
            if self.cfg.opt == "call":
                payoff = max(self.S - self.cfg.K, 0.0)
            else:
                payoff = max(self.cfg.K - self.S, 0.0)
            # close hedge at S with TC
            close_tc = self.cfg.tc_rate * abs(self.hedge) * self.S
            self.cash -= close_tc
            self.cum_tc += close_tc
            terminal_pnl = self.cash - payoff * self.cfg.notional
            # reward: squared terminal PnL (smaller variance = better)
            reward = -abs(terminal_pnl)
            info = {
                "terminal_pnl": terminal_pnl,
                "cum_tc": self.cum_tc,
                "cum_turnover": self.cum_hedge_turnover,
                "S_T": self.S,
            }
        else:
            reward = -0.01 * tc  # small per-step penalty encourages early convergence
            info = {}

        self.done = terminated
        return self._obs(), float(reward), terminated, truncated, info


__all__ = ["HedgingEnvConfig", "HedgingEnv"]
