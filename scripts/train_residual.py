"""B4 v4: Train residual hedger and compare to BSM Δ + v2 PPO."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_hedging" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))

from ai_hedging.env import HedgingEnvConfig
from ai_hedging.agents.residual_hedger import train_residual_ppo, evaluate_residual
from ai_hedging.baselines.bsm_delta import BSMDeltaHedger, rebalance_interval
from ai_hedging.reward import cvar_loss, sharpe


def stats(pnl):
    return dict(
        mean=float(pnl.mean()), std=float(pnl.std()),
        cvar=float(cvar_loss(pnl, 0.05)),
        sharpe=float(sharpe(pnl)),
    )


def main():
    tc = 0.003

    print("=== B4 Residual Hedger Training ===\n")

    eval_cfg = HedgingEnvConfig(
        tc_rate=tc, seed=0, reward_shaping=False,
        action_low=0.0, action_high=1.0,
    )

    # Baseline
    bsm = rebalance_interval(eval_cfg, every=1, hedger=BSMDeltaHedger("call"),
                             n_paths=2000, seed=777)
    bsm_st = stats(bsm["pnl_array"])
    print(f"[BSM Δ daily]    mean={bsm_st['mean']:+.4f} std={bsm_st['std']:.4f} "
          f"CVaR={bsm_st['cvar']:.4f} Sharpe={bsm_st['sharpe']:.4f}")

    # Train residual
    out = ROOT / "models" / "ppo_residual_tc03.zip"
    train_cfg = HedgingEnvConfig(
        tc_rate=tc, seed=0,
        reward_shaping=True,
        shaping_lambda=20.0,
        loss_penalty_mult=5.0,
        action_low=0.0, action_high=1.0,
    )
    t0 = time.time()
    train_residual_ppo(
        tc_rate=tc, total_timesteps=600_000, out=str(out),
        n_envs=8, seed=0, residual_bound=0.30,
        learning_rate=3e-4, ent_coef=0.005, device="cpu",
        env_cfg=train_cfg,
    )
    elapsed = time.time() - t0

    res = evaluate_residual(str(out), eval_cfg, n_paths=2000, seed=777,
                            residual_bound=0.30, device="cpu")
    res_st = stats(res["pnl_array"])
    print(f"[Residual PPO]   mean={res_st['mean']:+.4f} std={res_st['std']:.4f} "
          f"CVaR={res_st['cvar']:.4f} Sharpe={res_st['sharpe']:.4f}  "
          f"|residual|={res['mean_abs_residual']:.3f}")
    ratio = res_st["cvar"] / bsm_st["cvar"]
    improvement = (1 - ratio) * 100
    print(f"\n[ratio]  CVaR_residual / CVaR_BSM = {ratio:.3f}  "
          f"improvement = {improvement:+.1f}%  wall = {elapsed:.0f}s")

    if ratio < 0.80:
        verdict = "PASS (plan target 20%)"
    elif ratio < 1.0:
        verdict = "PARTIAL (improved over BSM)"
    else:
        verdict = "FAIL"
    print(f"[verdict] {verdict}")

    out_path = ROOT / "data" / "bench_ppo_residual.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"=== Residual Hedger TC={tc} ===\n")
        f.write(f"BSM Δ daily   CVaR={bsm_st['cvar']:.4f}\n")
        f.write(f"Residual PPO  CVaR={res_st['cvar']:.4f}  ratio={ratio:.3f}\n")
        f.write(f"improvement={improvement:.1f}%   |residual|={res['mean_abs_residual']:.3f}\n")
        f.write(f"verdict={verdict}\n")
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
