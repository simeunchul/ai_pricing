"""Train PPO v2 (tuned action + reward shaping) for both TC levels.

Writes:
  models/ppo_tc0_v2.zip
  models/ppo_tc03_v2.zip
  data/bench_ppo_v2.txt
"""
from __future__ import annotations

import argparse
import os
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
    equity = np.cumsum(pnl - pnl.mean())  # for max_dd proxy
    cum_max = np.maximum.accumulate(equity)
    dd = (equity - cum_max).min() if len(equity) else 0.0
    return dict(
        mean=float(pnl.mean()),
        std=float(pnl.std()),
        cvar=float(cvar_loss(pnl, 0.05)),
        sharpe=float(sharpe(pnl)),
        max_dd=float(dd),
    )


def run(tc: float, steps: int, out: str, n_envs: int, seed: int):
    cfg = HedgingEnvConfig(
        tc_rate=tc,
        seed=seed,
        reward_shaping=True,
        shaping_lambda=50.0,
        action_low=0.0,
        action_high=1.0,
    )
    t0 = time.time()
    train_ppo(
        tc_rate=tc,
        total_timesteps=steps,
        out=out,
        n_envs=n_envs,
        seed=seed,
        env_cfg=cfg,
        device="cpu",
    )
    elapsed = time.time() - t0
    # eval uses same cfg (reward_shaping flag doesn't affect terminal_pnl)
    eval_cfg = HedgingEnvConfig(
        tc_rate=tc,
        seed=0,
        reward_shaping=False,  # eval doesn't need shaping
        action_low=0.0,
        action_high=1.0,
    )
    res = evaluate_ppo(out, eval_cfg, n_paths=2000, seed=777, device="cpu")
    pnl = res["pnl_array"]
    st = pnl_stats(pnl)
    return st, elapsed


def bsm_bench(tc: float) -> dict:
    cfg = HedgingEnvConfig(tc_rate=tc, reward_shaping=False, seed=0)
    res = rebalance_interval(cfg, every=1, hedger=BSMDeltaHedger("call"),
                             n_paths=2000, seed=777)
    return pnl_stats(res["pnl_array"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-tc0", action="store_true")
    ap.add_argument("--skip-tc03", action="store_true")
    args = ap.parse_args()

    lines = []
    def log(s: str):
        print(s, flush=True)
        lines.append(s)

    log(f"=== PPO v2 training (steps={args.steps}, n_envs={args.n_envs}) ===")

    for tc, tag, out in [
        (0.0, "TC=0.000", ROOT / "models" / "ppo_tc0_v2.zip"),
        (0.003, "TC=0.003", ROOT / "models" / "ppo_tc03_v2.zip"),
    ]:
        if tc == 0.0 and args.skip_tc0:
            continue
        if tc == 0.003 and args.skip_tc03:
            continue
        log("")
        log(f"--- {tag} ---")
        bsm = bsm_bench(tc)
        log(f"[BSM]  mean={bsm['mean']:+.4f} std={bsm['std']:.4f} "
            f"CVaR5%={bsm['cvar']:.4f} Sharpe={bsm['sharpe']:.4f} maxDD={bsm['max_dd']:.4f}")

        st, elapsed = run(tc, args.steps, str(out), args.n_envs, args.seed)
        log(f"[PPO]  mean={st['mean']:+.4f} std={st['std']:.4f} "
            f"CVaR5%={st['cvar']:.4f} Sharpe={st['sharpe']:.4f} maxDD={st['max_dd']:.4f}")
        ratio = st["cvar"] / bsm["cvar"] if bsm["cvar"] > 0 else float("inf")
        log(f"[ratio] PPO_CVaR / BSM_CVaR = {ratio:.3f}  "
            f"(wall={elapsed:.1f}s, model={out.name})")

    bench_path = ROOT / "data" / "bench_ppo_v2.txt"
    bench_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[ok] wrote {bench_path}")


if __name__ == "__main__":
    main()
