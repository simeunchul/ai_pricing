"""Imitation + Residual learning for Deep Hedging.

핵심 아이디어 (sub-agent 의 §7 (c) 추천 경로):
  - BSM Δ 는 학습 안 하고 그대로 사용 (이미 이론적 베이스라인)
  - PPO 는 'residual' (BSM Δ 와의 차이) 만 학습
  - reward = -|terminal_pnl|  + cost_savings  (under-hedge 인센티브)

Why this works (vs vanilla PPO):
  - vanilla: PPO 가 [0,1] action 공간에서 BSM Δ 와 똑같은 policy 를
    배우려고 하는 attractor 에 끌림. TC>0 때 under-hedge 가 optimal 이지만
    학습 budget 부족하면 BSM 에 끌려감.
  - residual: action 자체를 (bsm_delta + δ) 로 정의. δ ∈ [-0.3, 0.3]
    의 좁은 박스에서 PPO 가 학습 → BSM 으로 안 끌려감.

이 구현은 Wrapper Env 로 감싸서 PPO 의 action 을 BSM-shifted 로 변환.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False

from pricing.bsm import BSMInputs
from pricing.greeks.analytic import call_greeks, put_greeks

from ai_hedging.env import HedgingEnv, HedgingEnvConfig


def bsm_delta_for_state(env: HedgingEnv) -> float:
    """Compute BSM Δ at current env state."""
    dt_rem = env.cfg.T * (1 - env.step_idx / env.cfg.n_steps)
    T_eff = max(dt_rem, 1e-6)
    inp = BSMInputs(env.S, env.cfg.K, T_eff, env.cfg.r, env.cfg.q, env.cfg.sigma)
    g = call_greeks(inp) if env.cfg.opt == "call" else put_greeks(inp)
    return float(g.delta) if env.cfg.opt == "call" else float(g.delta)


class ResidualHedgingEnv(gym.Env if _HAS_GYM else object):
    """Wrap HedgingEnv: agent emits residual δ, executed action = clip(BSM_Δ + δ, 0, 1).

    Action space: Box(-residual_bound, +residual_bound).
    Observation space: same as HedgingEnv (BSM Δ is internally derivable from obs).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: HedgingEnvConfig, residual_bound: float = 0.30):
        super().__init__()
        self._inner = HedgingEnv(cfg)
        self.cfg = cfg
        self.residual_bound = residual_bound

        self.action_space = spaces.Box(
            low=np.float32(-residual_bound),
            high=np.float32(residual_bound),
            shape=(1,), dtype=np.float32,
        )
        self.observation_space = self._inner.observation_space
        self._last_bsm = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = self._inner.reset(seed=seed, options=options)
        self._last_bsm = bsm_delta_for_state(self._inner)
        return obs, info

    def step(self, action):
        residual = float(np.clip(action, -self.residual_bound, self.residual_bound)[0])
        bsm_d = bsm_delta_for_state(self._inner)
        true_action = float(np.clip(bsm_d + residual, 0.0, 1.0))
        obs, reward, term, trunc, info = self._inner.step(np.array([true_action], dtype=np.float32))
        # Store residual into info for diagnostics
        info["residual"] = residual
        info["bsm_delta_used"] = bsm_d
        info["executed_action"] = true_action
        return obs, reward, term, trunc, info

    @property
    def step_idx(self):
        return self._inner.step_idx

    @property
    def S(self):
        return self._inner.S


def train_residual_ppo(
    tc_rate: float = 0.003,
    total_timesteps: int = 500_000,
    out: str = "models/ppo_residual.zip",
    n_envs: int = 8,
    seed: int = 0,
    residual_bound: float = 0.30,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.005,
    device: str = "cpu",
    env_cfg: HedgingEnvConfig | None = None,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = env_cfg or HedgingEnvConfig(
        tc_rate=tc_rate, seed=seed,
        reward_shaping=True,
        shaping_lambda=20.0,        # weak — let terminal pnl dominate
        loss_penalty_mult=5.0 if tc_rate > 0 else 1.0,
        action_low=0.0, action_high=1.0,
    )

    def make():
        return ResidualHedgingEnv(cfg, residual_bound=residual_bound)

    venv = DummyVecEnv([make for _ in range(n_envs)])
    model = PPO(
        "MlpPolicy", venv, verbose=1, seed=seed,
        learning_rate=learning_rate, n_steps=256, batch_size=256,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=ent_coef, device=device,
    )
    model.learn(total_timesteps=total_timesteps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return model


def evaluate_residual(model_path: str, env_cfg: HedgingEnvConfig,
                      n_paths: int = 2000, seed: int = 777,
                      residual_bound: float = 0.30, device: str = "cpu") -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device=device)
    env = ResidualHedgingEnv(env_cfg, residual_bound=residual_bound)
    pnls, residuals = [], []
    for p in range(n_paths):
        obs, _ = env.reset(seed=seed + p)
        done = False
        ep_residuals = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            ep_residuals.append(info.get("residual", 0.0))
            done = term or trunc
            if term:
                pnls.append(info["terminal_pnl"])
                residuals.append(np.mean(np.abs(ep_residuals)))
    arr = np.array(pnls)
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "pnl_array": arr,
        "mean_abs_residual": float(np.mean(residuals)),
    }


__all__ = ["ResidualHedgingEnv", "train_residual_ppo",
           "evaluate_residual", "bsm_delta_for_state"]
