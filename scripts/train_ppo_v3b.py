"""PPO v3b — minimal change from v2, just longer training + light entropy.

Hypothesis: sub-agent's v2 (ratio 0.92, 7.8% improvement) was bottlenecked by
~25-min CPU budget, not reward shaping. So this run keeps every reward knob
identical to v2 and only:
  - extends total_timesteps 500k → 1.5M
  - adds tiny entropy bonus (0.003) to keep exploration alive
  - same BC 1000 × 60ep warm-start

If this also fails to break 20%, it tells us the env reward design itself is
the bottleneck (not budget) and we'd need a different objective (e.g. CVaR
surrogate) — documented as honest finding rather than papered over.

Outputs:
  models/ppo_tc03_v3b.zip
  data/bench_ppo_v3b.txt
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
        mean=float(pnl.mean()), std=float(pnl.std()),
        cvar=float(cvar_loss(pnl, 0.05)),
        sharpe=float(sharpe(pnl)), max_dd=float(dd),
    )


def make_train_cfg(tc: float, seed: int) -> HedgingEnvConfig:
    """v2-identical reward shape (no pow, no tail_shock)."""
    return HedgingEnvConfig(
        tc_rate=tc, seed=seed,
        reward_shaping=True,
        shaping_lambda=50.0,             # v2 same
        action_low=0.0, action_high=1.0,
        loss_penalty_mult=5.0 if tc > 0 else 1.0,    # v2 same
        loss_penalty_pow=1.0,            # v2 default — no superlinear loss
        tail_shock_weight=0.0,           # v2 default — no per-step shock
    )


def make_eval_cfg(tc: float) -> HedgingEnvConfig:
    return HedgingEnvConfig(
        tc_rate=tc, seed=0, reward_shaping=False,
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
    ap.add_argument("--bc-epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    args = ap.parse_args()

    lines = []
    def log(s: str):
        print(s, flush=True); lines.append(s)

    log(f"=== PPO v3b — v2 reward + longer training ({args.steps:,} steps, "
        f"BC {args.bc_paths}x{args.bc_epochs}ep, lr={args.lr}, ent={args.ent_coef}) ===")

    tc = 0.003
    out = ROOT / "models" / "ppo_tc03_v3b.zip"

    log("")
    log(f"--- TC=0.003 ---")
    bsm = bsm_bench(tc)
    log(f"[BSM]  mean={bsm['mean']:+.4f} std={bsm['std']:.4f} "
        f"CVaR5%={bsm['cvar']:.4f} Sharpe={bsm['sharpe']:.4f} maxDD={bsm['max_dd']:.4f}")

    st, elapsed = run(tc, args.steps, str(out), args.n_envs, args.seed,
                      args.bc_paths, args.bc_epochs, args.lr, args.ent_coef)
    log(f"[PPO]  mean={st['mean']:+.4f} std={st['std']:.4f} "
        f"CVaR5%={st['cvar']:.4f} Sharpe={st['sharpe']:.4f} maxDD={st['max_dd']:.4f}")
    ratio = st["cvar"] / bsm["cvar"] if bsm["cvar"] > 0 else float("inf")
    improvement_pct = (1 - ratio) * 100
    verdict = "PASS" if ratio < 0.80 else ("PARTIAL" if ratio < 1.0 else "FAIL")
    log(f"[ratio] PPO_CVaR / BSM_CVaR = {ratio:.3f}  target<0.80  -> {verdict}  "
        f"(improvement={improvement_pct:.1f}%) wall={elapsed:.0f}s")

    bench_path = ROOT / "data" / "bench_ppo_v3b.txt"
    bench_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\n[ok] wrote {bench_path}")


if __name__ == "__main__":
    main()
