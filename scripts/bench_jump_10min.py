"""Jump 환경 + 10분 step — 가장 짧은 step 검증.

n_steps = 30 × 24 × 6 = 4,320 (calendar 기준 30일 × 24시간 × 6 (10분))
학습 cost ↑↑ → batch=1024, epoch=100 으로 조정 (약 15-20분 예상)

이전 1hour 결과 (+22.8%) 와 비교: 10분으로 더 짧아지면 NN 우위가 더 커지나? saturate 되나?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_hedging" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))

from ai_hedging.agents.buehler_pg import (
    BuehlerConfig, train_buehler, _bsm_call_premium,
)


def bsm_delta_baseline_jump(cfg: BuehlerConfig, n_paths: int, device="cpu", seed=777):
    import math
    from scipy.stats import norm

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    B = n_paths
    dt = cfg.T / cfg.n_steps
    drift_dt = (cfg.r - cfg.q - 0.5 * cfg.sigma ** 2) * dt
    if cfg.jump_intensity > 0.0 and cfg.jump_compensator:
        kappa = math.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
        drift_dt -= cfg.jump_intensity * kappa * dt
    diff_dt = cfg.sigma * math.sqrt(dt)
    exp_r_dt = math.exp(cfg.r * dt)
    lam_dt = cfg.jump_intensity * dt

    premium = _bsm_call_premium(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma)
    S = torch.full((B,), float(cfg.S0), device=device)
    cash = torch.full((B,), float(premium), device=device)
    hedge = torch.zeros(B, device=device)
    turnover = torch.zeros(B, device=device)

    for t in range(cfg.n_steps):
        T_rem = max(cfg.T * (1 - t / cfg.n_steps), 1e-6)
        d1 = (torch.log(S / cfg.K) + (cfg.r - cfg.q + 0.5 * cfg.sigma ** 2) * T_rem) \
             / (cfg.sigma * math.sqrt(T_rem))
        new_hedge = torch.tensor(norm.cdf(d1.cpu().numpy()), device=device, dtype=torch.float32)

        delta_h = torch.abs(new_hedge - hedge)
        tc = cfg.tc_rate * delta_h * S
        cash = cash - tc
        turnover = turnover + delta_h
        hedge = new_hedge

        Z = torch.randn(B, device=device, generator=gen)
        if lam_dt > 0.0:
            n_jumps = torch.poisson(torch.full((B,), lam_dt, device=device), generator=gen)
            jump_Z = torch.randn(B, device=device, generator=gen)
            log_jump = n_jumps * cfg.jump_mean + torch.sqrt(n_jumps) * cfg.jump_std * jump_Z
            S_next = S * torch.exp(drift_dt + diff_dt * Z + log_jump)
        else:
            S_next = S * torch.exp(drift_dt + diff_dt * Z)

        cash = cash * exp_r_dt + hedge * (S_next - S)
        S = S_next

    payoff = torch.relu(S - cfg.K)
    close_tc = cfg.tc_rate * torch.abs(hedge) * S
    cash = cash - close_tc
    pnl = (cash - payoff * cfg.notional).cpu().numpy()
    return {
        "pnl": pnl,
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "cvar_5": float(-np.sort(pnl)[:int(0.05 * len(pnl))].mean()),
        "turnover": float(turnover.mean().item()),
    }


def main():
    print("=" * 80)
    print("  Jump + 10분 step — n_steps=4320, batch=1024, epoch=100")
    print("=" * 80)

    jump_params = dict(
        jump_intensity=5.0,
        jump_mean=-0.02,
        jump_std=0.05,
        jump_compensator=True,
    )

    cfg = BuehlerConfig(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0, sigma=0.20,
        tc_rate=0.0010,
        notional=1.0, opt="call",
        n_steps=4320,            # 10분 step
        batch_size=1024,         # 메모리/속도 절충
        n_epochs=100,            # 절충 (이상적으론 200+)
        hidden=(128, 128, 64), activation="relu",
        lr=1e-3, cvar_alpha=0.05, grad_clip=1.0, seed=0,
        action_low=0.0, action_high=1.0,
        **jump_params,
    )

    print(f"[cfg] n_steps={cfg.n_steps}  batch={cfg.batch_size}  epochs={cfg.n_epochs}")
    print(f"      jump: λ={cfg.jump_intensity}/yr  μJ={cfg.jump_mean}  σJ={cfg.jump_std}")
    print(f"      TC={cfg.tc_rate*1e4:.0f}bps  σ_diffusion={cfg.sigma*100:.0f}%")
    print()

    out_path = str(ROOT / "models" / "buehler_jump_10min.pt")
    t0 = time.time()
    res = train_buehler(cfg, loss_type="cvar", out_path=out_path,
                        device="cpu", log_every=10, eval_batch=8000)
    elapsed = time.time() - t0

    cvar_b = res["final"]["cvar_5"]
    mean_b = res["final"]["mean_pnl"]
    std_b  = res["final"]["std_pnl"]
    print(f"\n[10min Buehler] CVaR={cvar_b:.4f}  mean={mean_b:+.4f}  std={std_b:.4f}  wall={elapsed:.0f}s")

    print("\n[BSM Δ at 10min step for baseline...]")
    bsm = bsm_delta_baseline_jump(cfg, n_paths=8000, seed=777)
    print(f"[BSM Δ 10min] CVaR={bsm['cvar_5']:.4f}  mean={bsm['mean']:+.4f}  "
          f"std={bsm['std']:.4f}  turnover={bsm['turnover']:.2f}")

    ratio = cvar_b / bsm["cvar_5"]
    improvement = (1 - ratio) * 100
    print()
    print("=" * 80)
    print(f"  RESULT — Jump + 10min step")
    print(f"  Buehler CVaR={cvar_b:.4f}  vs  BSM CVaR={bsm['cvar_5']:.4f}")
    print(f"  ratio={ratio:.3f}  improvement={improvement:+.1f}%")
    print("=" * 80)

    print("\n[Comparison vs previous step results in jump env]")
    print(f"  1day  : NN +14.1%  (BSM 3.96, NN 3.40)")
    print(f"  4hour : NN +17.4%  (BSM 4.09, NN 3.38)")
    print(f"  1hour : NN +22.8%  (BSM 4.45, NN 3.43)")
    print(f"  10min : NN {improvement:+.1f}%  (BSM {bsm['cvar_5']:.2f}, NN {cvar_b:.2f})")

    # save
    import json
    out_json = ROOT / "data" / "bench_jump_10min.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({
            "n_steps": cfg.n_steps,
            "epochs": cfg.n_epochs,
            "batch": cfg.batch_size,
            "wall_s": elapsed,
            "buehler_cvar": cvar_b,
            "buehler_mean": mean_b,
            "buehler_std": std_b,
            "bsm_cvar": bsm["cvar_5"],
            "bsm_mean": bsm["mean"],
            "bsm_turnover": bsm["turnover"],
            "ratio": ratio,
            "improvement_pct": improvement,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()
