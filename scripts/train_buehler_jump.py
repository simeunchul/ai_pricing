"""Buehler PG-on-CVaR 를 jump 환경에서 재학습.

핵심: 정책 입력은 5-dim 그대로 (jump 신호 0개) — Buehler model-free 정신 유지.
변경되는 건 *학습 환경* 의 분포뿐.

두 모드:
  --mode fixed   : λ=10 jumps/year 고정 환경에서 학습
  --mode random  : 매 epoch 마다 λ ~ Uniform(0, 18) 샘플 (domain randomization)

Outputs
-------
fixed  → models/buehler_jump_fixed.pt
random → models/buehler_jump_random.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_hedging" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))

from ai_hedging.agents.buehler_pg import (
    BuehlerConfig, HedgingPolicy, simulate_batch, cvar_loss_fn,
)


def base_cfg() -> BuehlerConfig:
    """Same config as the existing buehler_tc03_v2 for fair compare."""
    return BuehlerConfig(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0, sigma=0.20,
        n_steps=30, tc_rate=0.003,
        hidden=(128, 128, 64), activation="relu",
        batch_size=16384, lr=1e-3, n_epochs=600,
        cvar_alpha=0.05, grad_clip=1.0, seed=0,
        action_low=0.0, action_high=1.0,
    )


def sample_jump_params(rng: np.random.Generator) -> tuple[float, float, float]:
    """Domain randomization sampler.

    Range chosen to (a) cover most of the test distribution but (b) leave
    'Large jump' (λ=20) slightly OOD so we still measure generalization.
    """
    lam  = float(rng.uniform(0.0, 18.0))
    muJ  = -0.01 - 0.003 * (lam / 18.0) * 7.0      # 0 → -0.01 ;  18 → -0.031
    sigJ = 0.03 + 0.005 * lam                      # 0 → 0.03 ;   18 → 0.12
    return lam, muJ, sigJ


def train_loop(cfg: BuehlerConfig, mode: str, out_path: str,
               device: str = "cpu", log_every: int = 20) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng_dr = np.random.default_rng(cfg.seed + 1)

    policy = HedgingPolicy(
        hidden=cfg.hidden, activation=cfg.activation,
        action_low=cfg.action_low, action_high=cfg.action_high,
    ).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    history = {"epoch": [], "cvar": [], "mean_pnl": [], "std_pnl": [],
               "lam": [], "muJ": [], "sigJ": []}

    for ep in range(cfg.n_epochs):
        policy.train()

        if mode == "fixed":
            # λ=10 고정. cfg 자체에 이미 jump 가 박혀 있다고 가정.
            cfg_ep = cfg
        elif mode == "random":
            lam, muJ, sigJ = sample_jump_params(rng_dr)
            cfg_ep = BuehlerConfig(**{
                **cfg.__dict__,
                "jump_intensity": lam,
                "jump_mean": muJ,
                "jump_std": sigJ,
            })
        else:
            raise ValueError(mode)

        pnl = simulate_batch(policy, cfg_ep, cfg.batch_size, device=device,
                             seed=cfg.seed + ep)
        loss = cvar_loss_fn(pnl, cfg.cvar_alpha)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        with torch.no_grad():
            mean_pnl = pnl.mean().item()
            std_pnl  = pnl.std().item()
            cvar     = cvar_loss_fn(pnl, cfg.cvar_alpha).item()
        history["epoch"].append(ep)
        history["cvar"].append(cvar)
        history["mean_pnl"].append(mean_pnl)
        history["std_pnl"].append(std_pnl)
        history["lam"].append(cfg_ep.jump_intensity)
        history["muJ"].append(cfg_ep.jump_mean)
        history["sigJ"].append(cfg_ep.jump_std)

        if ep % log_every == 0 or ep == cfg.n_epochs - 1:
            print(f"[ep {ep:3d}/{cfg.n_epochs}] loss={loss.item():+.4f}  "
                  f"CVaR={cvar:.4f}  mean={mean_pnl:+.4f}  "
                  f"λ={cfg_ep.jump_intensity:.1f}", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # 저장 cfg 는 'base' 만 (jump 파라미터는 학습 시 변동했으므로
    # checkpoint 의 cfg 는 시나리오 평가 시 무관). Stress test 가 자체 시나리오 cfg 를 만듦.
    save_cfg = {**cfg.__dict__,
                "_train_mode": mode,
                # fixed 모드면 학습 jump 파라미터가 cfg 에 남아있음
                # random 모드면 0 으로 리셋해서 저장 (참고용)
                **({"jump_intensity": 0.0, "jump_mean": 0.0, "jump_std": 0.0}
                   if mode == "random" else {})}
    torch.save({"state_dict": policy.state_dict(), "cfg": save_cfg,
                "history_summary": {
                    "n_epochs": cfg.n_epochs,
                    "final_cvar": history["cvar"][-1],
                    "mode": mode,
                }}, out_path)
    return {"history": history, "model_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "random"], required=True)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = base_cfg()
    cfg.n_epochs = args.epochs

    if args.mode == "fixed":
        # λ=10, μJ=-0.05, σJ=0.10 — Medium jump (test 의 medium 시나리오와 동일)
        cfg = BuehlerConfig(**{
            **cfg.__dict__,
            "jump_intensity": 10.0,
            "jump_mean": -0.05,
            "jump_std": 0.10,
        })
        out = args.out or str(ROOT / "models" / "buehler_jump_fixed.pt")
        print("=" * 70)
        print(f"  Buehler PG-CVaR — FIXED jump training (λ=10, μJ=-0.05, σJ=0.10)")
    else:
        out = args.out or str(ROOT / "models" / "buehler_jump_random.pt")
        print("=" * 70)
        print(f"  Buehler PG-CVaR — DOMAIN-RANDOM jump training (λ ~ U(0,18))")
    print(f"  out={out}  epochs={cfg.n_epochs}  batch={cfg.batch_size}")
    print("=" * 70)
    print()

    t0 = time.time()
    res = train_loop(cfg, args.mode, out)
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print(f"  done in {elapsed:.0f}s  →  {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
