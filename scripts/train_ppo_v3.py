"""PPO v3 — push TC=0.003 CVaR improvement past plan target (20%).

Changes vs v2:
  - loss_penalty_pow=1.5  (terminal: -|pnl|^1.5 * mult when pnl<0)
                          → large-loss gradient dominates → tail-aware
  - tail_shock_weight    (per-step extra penalty on |rep_err| > thr)
  - Larger BC pretrain (1000 paths × 80 epochs)
  - PPO 1.5M steps with smaller LR (5e-5) and tiny entropy (0.003)

Outputs:
  models/ppo_tc0_v3.zip
  models/ppo_tc03_v3.zip
  data/bench_ppo_v3.txt

Usage:
  python scripts/train_ppo_v3.py
  python scripts/train_ppo_v3.py --skip-tc0    # only TC=0.003 (faster)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_hedging" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))

from ai_hedging.env import HedgingEnv, HedgingEnvConfig
from ai_hedging.agents.ppo_hedger import train_ppo, evaluate_ppo
from ai_hedging.baselines.bsm_delta import BSMDeltaHedger, rebalance_interval
from ai_hedging.reward import cvar_loss, sharpe


def pnl_stats(pnl: np.ndarray) -> dict:
    equity = np.cumsum(pnl - pnl.mean())
    cum_max = np.maximum.accumulate(equity)
    dd = (equity - cum_max).min() if len(equity) else 0.0
    return dict(
        mean=float(pnl.mean()),
        std=float(pnl.std()),
        cvar=float(cvar_loss(pnl, 0.05)),
        sharpe=float(sharpe(pnl)),
        max_dd=float(dd),
    )


def make_train_cfg(tc: float, seed: int) -> HedgingEnvConfig:
    """Training config: dense shaping + tail penalty + amplified-loss terminal."""
    return HedgingEnvConfig(
        tc_rate=tc,
        seed=seed,
        reward_shaping=True,
        shaping_lambda=80.0,           # ↑ from 50
        action_low=0.0,
        action_high=1.0,
        loss_penalty_mult=8.0 if tc > 0 else 1.0,    # ↑ from 5
        loss_penalty_pow=1.5 if tc > 0 else 1.0,     # NEW — superlinear loss
        tail_shock_weight=10.0,        # NEW — per-step shock penalty
        tail_shock_thr=0.04,
    )


def make_eval_cfg(tc: float) -> HedgingEnvConfig:
    """Eval config: NO shaping (eval terminal_pnl is what matters)."""
    return HedgingEnvConfig(
        tc_rate=tc, seed=0,
        reward_shaping=False,
        action_low=0.0, action_high=1.0,
    )


def run(tc: float, steps: int, out: str, n_envs: int, seed: int,
        bc_paths: int, bc_epochs: int, lr: float, ent_coef: float):
    cfg = make_train_cfg(tc, seed)
    t0 = time.time()
    train_ppo(
        tc_rate=tc, total_timesteps=steps, out=out,
        n_envs=n_envs, seed=seed, env_cfg=cfg, device="cpu",
        bc_warmstart_epochs=bc_epochs, bc_warmstart_paths=bc_paths,
        learning_rate=lr, ent_coef=ent_coef,
    )
    elapsed = time.time() - t0
    res = evaluate_ppo(out, make_eval_cfg(tc), n_paths=2000, seed=777, device="cpu")
    return pnl_stats(res["pnl_array"]), elapsed


def bsm_bench(tc: float) -> dict:
    res = rebalance_interval(make_eval_cfg(tc), every=1,
                             hedger=BSMDeltaHedger("call"),
                             n_paths=2000, seed=777)
    return pnl_stats(res["pnl_array"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bc-paths", type=int, default=1000)
    ap.add_argument("--bc-epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--skip-tc0", action="store_true")
    ap.add_argument("--skip-tc03", action="store_true")
    args = ap.parse_args()

    lines = []
    def log(s: str):
        print(s, flush=True); lines.append(s)

    log(f"=== PPO v3 (steps={args.steps:,} bc={args.bc_paths}x{args.bc_epochs}ep "
        f"lr={args.lr} ent={args.ent_coef} n_envs={args.n_envs}) ===")

    targets = []
    if not args.skip_tc0:
        targets.append((0.0,   "TC=0.000", ROOT / "models" / "ppo_tc0_v3.zip", 1.10))
    if not args.skip_tc03:
        targets.append((0.003, "TC=0.003", ROOT / "models" / "ppo_tc03_v3.zip", 0.80))

    for tc, tag, out, target_ratio in targets:
        log("")
        log(f"--- {tag} ---")
        bsm = bsm_bench(tc)
        log(f"[BSM]  mean={bsm['mean']:+.4f} std={bsm['std']:.4f} "
            f"CVaR5%={bsm['cvar']:.4f} Sharpe={bsm['sharpe']:.4f} maxDD={bsm['max_dd']:.4f}")

        st, elapsed = run(tc, args.steps, str(out), args.n_envs, args.seed,
                          args.bc_paths, args.bc_epochs, args.lr, args.ent_coef)
        log(f"[PPO]  mean={st['mean']:+.4f} std={st['std']:.4f} "
            f"CVaR5%={st['cvar']:.4f} Sharpe={st['sharpe']:.4f} maxDD={st['max_dd']:.4f}")
        ratio = st["cvar"] / bsm["cvar"] if bsm["cvar"] > 0 else float("inf")
        improvement_pct = (1 - ratio) * 100 if tc > 0 else 0
        verdict = "PASS" if ratio < target_ratio else ("PARTIAL" if ratio < 1.0 else "FAIL")
        log(f"[ratio] PPO_CVaR / BSM_CVaR = {ratio:.3f}  "
            f"target<{target_ratio}  -> {verdict}  "
            f"({'sanity' if tc==0 else f'improvement={improvement_pct:.1f}%'})  "
            f"wall={elapsed:.0f}s")

    bench_path = ROOT / "data" / "bench_ppo_v3.txt"
    bench_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n[ok] wrote {bench_path}")


if __name__ == "__main__":
    main()
