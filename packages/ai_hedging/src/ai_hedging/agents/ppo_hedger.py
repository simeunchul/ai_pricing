"""PPO agent for Deep Hedging (Buehler et al. 2019 style).

Usage:
    python -m ai_hedging.agents.ppo_hedger --tc 0.003 --steps 200000 --out models/ppo_tc03.zip

v2 additions:
    - optional supervised warm-start on BSM Δ (bc_warmstart_epochs > 0).
    - tuned hyperparams for narrow action box [0, 1].
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_hedging.env import HedgingEnv, HedgingEnvConfig


def _bc_warmstart(model, env_cfg: HedgingEnvConfig, n_paths: int = 200,
                  epochs: int = 5, lr: float = 1e-3, verbose: bool = False):
    """Behavior cloning warm-start: pre-train policy to mimic BSM Δ on random paths.

    Only touches the policy's action head. Value function is left alone.
    """
    import torch
    from ai_hedging.env import HedgingEnv
    from ai_hedging.baselines.bsm_delta import BSMDeltaHedger

    dev = model.device
    env = HedgingEnv(env_cfg)
    hedger = BSMDeltaHedger("call" if env_cfg.opt == "call" else "put")

    xs, ys = [], []
    for p in range(n_paths):
        obs, _ = env.reset(seed=1_000_000 + p)
        done = False
        while not done:
            target = hedger.act(env)
            target = float(np.clip(target, env_cfg.action_low, env_cfg.action_high))
            xs.append(obs)
            ys.append([target])
            obs, _, term, trunc, _ = env.step(np.array([target], dtype=np.float32))
            done = term or trunc

    X = torch.as_tensor(np.array(xs), dtype=torch.float32, device=dev)
    Y = torch.as_tensor(np.array(ys), dtype=torch.float32, device=dev)

    policy = model.policy
    # only optimize the actor (policy_net + action_net); skip value net
    params = list(policy.mlp_extractor.policy_net.parameters()) + \
             list(policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    bs = 512
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = X[idx], Y[idx]
            feats = policy.extract_features(xb)
            if isinstance(feats, tuple):
                feats_pi, _ = feats
            else:
                feats_pi = feats
            latent_pi = policy.mlp_extractor.forward_actor(feats_pi)
            mean = policy.action_net(latent_pi)
            # map tanh-squashed Gaussian mean roughly into action range via clamp
            # PPO MlpPolicy uses DiagGaussianDistribution (no squashing) — mean is raw
            loss = ((mean - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if verbose:
            print(f"[warmstart] epoch {ep+1}/{epochs} mse={ep_loss/max(nb,1):.6f}", flush=True)
    return model


def train_ppo(
    tc_rate: float = 0.0,
    total_timesteps: int = 100_000,
    out: str = "models/ppo_hedger.zip",
    n_envs: int = 4,
    seed: int = 0,
    env_cfg: HedgingEnvConfig | None = None,
    device: str = "cpu",
    bc_warmstart_epochs: int = 0,
    bc_warmstart_paths: int = 200,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.0,
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
        learning_rate=learning_rate,
        n_steps=256,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        device=device,   # "cpu" default protects shared VRAM; MlpPolicy is CPU-friendly anyway
    )
    if bc_warmstart_epochs > 0:
        _bc_warmstart(model, cfg, n_paths=bc_warmstart_paths,
                      epochs=bc_warmstart_epochs, lr=1e-3, verbose=True)
    model.learn(total_timesteps=total_timesteps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return model


def evaluate_ppo(model_path: str, env_cfg: HedgingEnvConfig, n_paths: int = 2000,
                 seed: int = 123, device: str = "cpu") -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device=device)
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
