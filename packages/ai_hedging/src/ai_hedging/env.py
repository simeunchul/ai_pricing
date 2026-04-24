"""Gym(nasium) environment for Deep Hedging.

We hedge a short European call under GBM with proportional transaction cost.
Episode: one option lifetime discretized into `n_steps`.

State (normalized, dim 5):
    [log(S/K), T_remaining/T_total, current_hedge, sigma*sqrt(T_rem), cash_norm]

Action (v2): Box([0, 1]) — desired hedge (call delta domain).
    Tight box prevents PPO from exploring economically absurd regions;
    calls have delta in [0, 1], puts in [-1, 0] (see `opt`).

Reward (v2, Buehler 2019 eq.13 style dense shaping):
    - per-step: -lam * (dV_portfolio - dV_option)^2 - tc_penalty
      where dV_portfolio = hedge * dS + cash*(e^{r dt}-1) - tc
            dV_option    = BSM_call(t+1) - BSM_call(t)  (reference, training-only)
      This makes replication error dense and scalar; with lam>0 PPO directly
      minimizes per-step hedging variance instead of waiting until T.
    - terminal: -|terminal_pnl| (kept, but now dominated by shaped steps).

The BSM reference is a *reward* signal only. At inference time the agent sees
the same 5-dim state and does not query BSM.
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

from pricing.bsm import BSMInputs, call_price, put_price


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
    # v2 reward shaping knobs ------------------------------------------------
    reward_shaping: bool = True       # dense -(dV_hedged - dV_opt)^2 penalty
    shaping_lambda: float = 50.0      # weight on the squared replication error
    action_low: float = 0.0           # call delta ∈ [0,1]; override for puts
    action_high: float = 1.0
    # CVaR-oriented terminal reward weights
    loss_penalty_mult: float = 1.0    # extra multiplier on |terminal_pnl| when pnl<0


class HedgingEnv(gym.Env if _HAS_GYM else object):
    """Gym env. Short 1 call, choose hedge each step. Reward = -per-step PnL variance."""

    metadata = {"render_modes": []} if _HAS_GYM else {}

    def __init__(self, cfg: HedgingEnvConfig | None = None):
        if not _HAS_GYM:
            raise ImportError("pip install gymnasium stable-baselines3 torch")
        super().__init__()
        self.cfg = cfg or HedgingEnvConfig()
        self.action_space = spaces.Box(
            low=np.float32(self.cfg.action_low),
            high=np.float32(self.cfg.action_high),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32,
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self._reset_state()

    # -- BSM reference option value (used for reward shaping only) -----------
    def _opt_price(self, S: float, t_rem: float) -> float:
        t_eff = max(t_rem, 1e-8)
        inp = BSMInputs(S, self.cfg.K, t_eff, self.cfg.r, self.cfg.q, self.cfg.sigma)
        return call_price(inp) if self.cfg.opt == "call" else put_price(inp)

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

        # snapshot pre-step option reference value (for reward shaping)
        dt = self.cfg.T / self.cfg.n_steps
        t_rem_pre = self.cfg.T * (1 - self.step_idx / self.cfg.n_steps)
        V_opt_pre = self._opt_price(self.S, t_rem_pre) if self.cfg.reward_shaping else 0.0
        S_pre = self.S
        cash_pre = self.cash

        # advance underlying one step (GBM)
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

        # -- dense shaping reward (Buehler 2019 eq.13 approximation) ---------
        if self.cfg.reward_shaping and not terminated:
            t_rem_post = self.cfg.T * (1 - self.step_idx / self.cfg.n_steps)
            V_opt_post = self._opt_price(self.S, t_rem_post)
            # portfolio value change = cash gain + hedge gain (we already updated cash)
            dV_hedged = (self.cash - cash_pre)
            # short-call P&L: -(V_opt_post - V_opt_pre) + dV_hedged
            # Replication error per step (we want this to be zero):
            rep_err = dV_hedged - (V_opt_post - V_opt_pre)
            shape_reward = -self.cfg.shaping_lambda * (rep_err * rep_err)
        else:
            shape_reward = 0.0

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
            # CVaR-aware: amplify losses (pnl<0) by loss_penalty_mult
            if terminal_pnl < 0:
                term_reward = -abs(terminal_pnl) * self.cfg.loss_penalty_mult
            else:
                term_reward = -abs(terminal_pnl)
            reward = term_reward + shape_reward
            info = {
                "terminal_pnl": terminal_pnl,
                "cum_tc": self.cum_tc,
                "cum_turnover": self.cum_hedge_turnover,
                "S_T": self.S,
            }
        else:
            # small tc penalty retained for turnover discouragement
            reward = shape_reward - 0.01 * tc
            info = {}

        self.done = terminated
        return self._obs(), float(reward), terminated, truncated, info


__all__ = ["HedgingEnvConfig", "HedgingEnv"]
