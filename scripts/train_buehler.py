"""Train Buehler 2019 PG-on-CVaR + compare to BSM Δ baseline.

Outputs:
  models/buehler_tc03.pt
  data/bench_buehler.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_hedging" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))

from ai_hedging.agents.buehler_pg import (
    BuehlerConfig, train_buehler, evaluate, simulate_batch, HedgingPolicy,
)
from ai_hedging.env import HedgingEnv, HedgingEnvConfig
from ai_hedging.baselines.bsm_delta import BSMDeltaHedger, rebalance_interval
from ai_hedging.reward import cvar_loss


def bsm_baseline(tc: float, n_paths: int = 8000):
    """Use existing gym-based BSM Δ for fair compare."""
    cfg = HedgingEnvConfig(tc_rate=tc, seed=0, reward_shaping=False,
                            action_low=0.0, action_high=1.0)
    res = rebalance_interval(cfg, every=1, hedger=BSMDeltaHedger("call"),
                             n_paths=n_paths, seed=777)
    return res["pnl_array"]


def main():
    print("=" * 70)
    print("  Buehler 2019 — Direct PG on CVaR loss")
    print("=" * 70)

    # Same env config as previous PPO experiments for fair comparison
    cfg = BuehlerConfig(
        S0=100.0, K=100.0, T=30 / 365, r=0.02, q=0.0, sigma=0.20,
        n_steps=30, tc_rate=0.003,
        batch_size=4096, lr=1e-3, n_epochs=200,
        cvar_alpha=0.05, grad_clip=1.0, seed=0,
        action_low=0.0, action_high=1.0,
    )

    out = str(ROOT / "models" / "buehler_tc03.pt")

    print(f"[cfg] tc={cfg.tc_rate}  T={cfg.T:.4f}y  n_steps={cfg.n_steps}  "
          f"batch={cfg.batch_size}  epochs={cfg.n_epochs}  α={cfg.cvar_alpha}")
    print()

    t0 = time.time()
    res = train_buehler(cfg, loss_type="cvar", out_path=out,
                         device="cpu", log_every=10, eval_batch=8000)
    elapsed = time.time() - t0

    buehler_pnl = res["eval_pnl"]
    buehler_cvar = res["final"]["cvar_5"]
    print(f"\n[Buehler PG-CVaR]  mean={buehler_pnl.mean():+.4f}  "
          f"std={buehler_pnl.std():.4f}  CVaR={buehler_cvar:.4f}  "
          f"wall={elapsed:.0f}s")

    # BSM baseline (gym env, same config)
    bsm_pnl = bsm_baseline(cfg.tc_rate, n_paths=8000)
    bsm_cvar = float(cvar_loss(bsm_pnl, 0.05))
    print(f"[BSM Δ daily]      mean={bsm_pnl.mean():+.4f}  "
          f"std={bsm_pnl.std():.4f}  CVaR={bsm_cvar:.4f}")

    ratio = buehler_cvar / bsm_cvar
    improvement = (1 - ratio) * 100
    if ratio < 0.80:
        verdict = "PASS (plan target ≥20% improvement)"
    elif ratio < 1.0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print()
    print("=" * 70)
    print(f"  CVaR ratio = {ratio:.3f}    improvement = {improvement:+.1f}%")
    print(f"  Verdict: {verdict}")
    print("=" * 70)

    log_path = ROOT / "data" / "bench_buehler.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== Buehler 2019 PG-on-CVaR (TC={cfg.tc_rate}) ===\n")
        f.write(f"Wall: {elapsed:.0f}s, batch={cfg.batch_size}, "
                f"epochs={cfg.n_epochs}, alpha={cfg.cvar_alpha}\n\n")
        f.write(f"[Buehler]  mean={buehler_pnl.mean():+.4f}  "
                f"std={buehler_pnl.std():.4f}  CVaR={buehler_cvar:.4f}\n")
        f.write(f"[BSM Δ]    mean={bsm_pnl.mean():+.4f}  "
                f"std={bsm_pnl.std():.4f}  CVaR={bsm_cvar:.4f}\n\n")
        f.write(f"ratio = {ratio:.3f}    improvement = {improvement:+.1f}%\n")
        f.write(f"verdict = {verdict}\n\n")
        # learning curve summary (every 20 epochs)
        f.write("learning curve (every 20 ep):\n")
        h = res["history"]
        for i in range(0, len(h["epoch"]), 20):
            f.write(f"  ep {h['epoch'][i]:3d}: CVaR={h['cvar'][i]:+.4f} "
                    f"mean={h['mean_pnl'][i]:+.4f}  std={h['std_pnl'][i]:.4f}\n")
    print(f"\n→ {log_path}")


if __name__ == "__main__":
    main()
