"""PPO agent for Deep Hedging (Buehler et al. 2019 style).

Usage:
    python -m ai_hedging.agents.ppo_hedger --tc 0.003 --steps 200000 --out models/ppo_tc03.zip
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_hedging.env import HedgingEnv, HedgingEnvConfig


def train_ppo(
    tc_rate: float = 0.0,
    total_timesteps: int = 100_000,
    out: str = "models/ppo_hedger.zip",
    n_envs: int = 4,
    seed: int = 0,
    env_cfg: HedgingEnvConfig | None = None,
):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        raise ImportError("pip install stable-baselines3 gymnasium torch") from e

    cfg = env_cfg or HedgingEnvConfig(tc_rate=tc_rate, seed=seed)

    def make():
        return HedgingEnv(cfg)

    venv = DummyVecEnv([make for _ in range(n_envs)])
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    model.learn(total_timesteps=total_timesteps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return model


def evaluate_ppo(model_path: str, env_cfg: HedgingEnvConfig, n_paths: int = 2000,
                 seed: int = 123) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = HedgingEnv(env_cfg)
    pnls = []
    for p in range(n_paths):
        obs, _ = env.reset(seed=seed + p)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if terminated:
                pnls.append(info["terminal_pnl"])
    arr = np.array(pnls)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.quantile(arr, 0.05)),
        "pnl_array": arr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", type=float, default=0.003)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--out", type=str, default="models/ppo_hedger.zip")
    args = ap.parse_args()
    train_ppo(tc_rate=args.tc, total_timesteps=args.steps, out=args.out, n_envs=args.n_envs)


if __name__ == "__main__":
    main()
