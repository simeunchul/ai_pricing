"""Jump 환경 + step 단축 비교 — 가설 검증.

가설: BSM Δ 는 정규분포 가정이라 jump 못 잡음.
       NN 은 1시간 step + 충분 학습 budget 이면 jump 노출 시간 ↓ → BSM 우위 확대 예상.

이전 GBM 비교 (TC=10bps, epoch 부족) 에서는 NN 이 짧은 step 에서 패배했음.
이번엔 (1) jump 추가 (2) 200 epoch 통일 (3) TC=10bps 유지.
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
    BuehlerConfig, HedgingPolicy, simulate_batch, train_buehler,
    _bsm_call_premium,
)


def bsm_delta_baseline_jump(cfg: BuehlerConfig, n_paths: int, device="cpu", seed=777):
    """BSM Δ on the SAME jump-diffusion paths (BSM ignores jump in delta calc)."""
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
    print("  Jump 환경 + Step 단축 — 가설 검증 (NN 우위가 jump 환경에서 발현?)")
    print("=" * 80)

    # Jump 시나리오: λ=5/yr (적당), μJ=-0.02 (평균 -2% 점프), σJ=0.05
    # 30일 만기 동안 평균 5×30/365 = 0.41 번 점프 (50% 확률로 점프 1회 이상)
    jump_params = dict(
        jump_intensity=5.0,
        jump_mean=-0.02,
        jump_std=0.05,
        jump_compensator=True,
    )

    # 모든 케이스 200 epoch 통일 (이전 실험의 학습 부족 문제 해결)
    scenarios = [
        ("1day",  30,   200, 4096, "baseline"),
        ("8hour", 90,   200, 4096, "3× 거래"),
        ("4hour", 180,  200, 4096, "6× 거래"),
        ("1hour", 720,  200, 2048, "24× 거래, 12분 예상"),
    ]

    base_cfg = dict(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0, sigma=0.20,
        tc_rate=0.0010,
        notional=1.0, opt="call",
        hidden=(128, 128, 64), activation="relu",
        lr=1e-3, cvar_alpha=0.05, grad_clip=1.0, seed=0,
        action_low=0.0, action_high=1.0,
        **jump_params,
    )

    results = []
    for name, n_steps, epochs, batch, note in scenarios:
        print(f"\n{'='*80}")
        print(f"  Scenario: {name} (n_steps={n_steps}, epochs={epochs}, batch={batch}) — {note}")
        print(f"{'='*80}")

        cfg = BuehlerConfig(
            **base_cfg,
            n_steps=n_steps,
            batch_size=batch,
            n_epochs=epochs,
        )
        out_path = str(ROOT / "models" / f"buehler_jump_step_{name}.pt")
        t0 = time.time()
        res = train_buehler(cfg, loss_type="cvar", out_path=out_path,
                            device="cpu", log_every=max(epochs // 5, 1),
                            eval_batch=8000)
        elapsed = time.time() - t0

        pnl_buehler = res["eval_pnl"]
        cvar_b = res["final"]["cvar_5"]
        mean_b = res["final"]["mean_pnl"]
        std_b = res["final"]["std_pnl"]

        print(f"\n[BSM Δ at same step ({name}) for baseline...]")
        bsm = bsm_delta_baseline_jump(cfg, n_paths=8000, seed=777)
        cvar_bsm = bsm["cvar_5"]
        mean_bsm = bsm["mean"]
        ratio = cvar_b / cvar_bsm
        improvement = (1 - ratio) * 100

        result = {
            "name": name, "n_steps": n_steps, "epochs": epochs,
            "wall_s": elapsed,
            "buehler_cvar": cvar_b, "buehler_mean": mean_b, "buehler_std": std_b,
            "bsm_cvar": cvar_bsm, "bsm_mean": mean_bsm, "bsm_turnover": bsm["turnover"],
            "ratio": ratio, "improvement_pct": improvement,
        }
        results.append(result)
        print(f"\n[{name}] Buehler CVaR={cvar_b:.4f}  BSM CVaR={cvar_bsm:.4f}  "
              f"ratio={ratio:.3f}  improvement={improvement:+.1f}%  "
              f"wall={elapsed:.0f}s  BSM turnover={bsm['turnover']:.2f}")

    print("\n" + "=" * 80)
    print("  SUMMARY — Jump 환경 step size 효과")
    print("=" * 80)
    print(f"{'Scenario':<10} {'n_steps':>8} {'BSM CVaR':>10} {'NN CVaR':>10} "
          f"{'ratio':>7} {'+%':>7} {'wall':>7} {'BSM turn':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<10} {r['n_steps']:>8} {r['bsm_cvar']:>10.4f} "
              f"{r['buehler_cvar']:>10.4f} {r['ratio']:>7.3f} "
              f"{r['improvement_pct']:>+6.1f}% {r['wall_s']:>6.0f}s "
              f"{r['bsm_turnover']:>10.2f}")

    import json
    out_json = ROOT / "data" / "bench_jump_step.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()
