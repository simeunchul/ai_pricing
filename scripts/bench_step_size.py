"""Step size 효과 검증 — Buehler PG-on-CVaR 를 1일/8시간/4시간/1시간 step 으로 학습.

핵심 가설: step 짧을수록 위험 노출 시간 ↓ → CVaR ↓
반대 효과: 거래 횟수 ↑ → 누적 TC ↑

비교 환경: 30일 ATM 콜, GBM (σ=20%), TC=10bps (proportional)
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
    _bsm_call_premium, cvar_loss_fn,
)


def bsm_delta_baseline(cfg: BuehlerConfig, n_paths: int, device="cpu", seed=777):
    """BSM Δ daily-style on the SAME GBM paths at the same step."""
    import math
    from scipy.stats import norm

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    B = n_paths
    dt = cfg.T / cfg.n_steps
    drift_dt = (cfg.r - cfg.q - 0.5 * cfg.sigma ** 2) * dt
    diff_dt = cfg.sigma * math.sqrt(dt)
    exp_r_dt = math.exp(cfg.r * dt)

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
    print("  Step Size Bench — Buehler vs BSM Δ at 1d / 8h / 4h / 1h")
    print("=" * 80)

    # 30 days, 24×6.5 trading hours/day = 156h ≈ 30×156=4680 hours over 30d if 24h
    # We use trading hours convention: 6.5 hours/day, so 30d = 195 hours
    # Step counts:
    #   1 day  = 30 steps
    #   8 hour = 30 × (24/8) = 90  (calendar) — use 30×3 = 90
    #   4 hour = 30 × 6 = 180
    #   1 hour = 30 × 24 = 720      (calendar)
    # For simplicity use calendar hours (realistic enough for relative comparison)

    scenarios = [
        ("1day",  30,   200, 4096, "baseline (학술 표준)"),
        ("8hour", 90,   100, 4096, "3× 거래"),
        ("4hour", 180,  100, 4096, "6× 거래"),
        ("1hour", 720,  60,  2048, "24× 거래"),
    ]

    # Same GBM environment (30 day ATM call), TC=10bps (intraday-realistic)
    # Daily 30bps was for daily — for short steps we drop to 10bps to model
    # tighter spreads at higher freq (still single proportional model)
    base_cfg = dict(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0, sigma=0.20,
        tc_rate=0.0010,   # 10bps proportional
        notional=1.0, opt="call",
        hidden=(128, 128, 64), activation="relu",
        lr=1e-3, cvar_alpha=0.05, grad_clip=1.0, seed=0,
        action_low=0.0, action_high=1.0,
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
        out_path = str(ROOT / "models" / f"buehler_step_{name}.pt")
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
        bsm = bsm_delta_baseline(cfg, n_paths=8000, seed=777)
        cvar_bsm = bsm["cvar_5"]
        mean_bsm = bsm["mean"]
        ratio = cvar_b / cvar_bsm
        improvement = (1 - ratio) * 100

        # turnover from buehler model — re-run eval to count rebalance volume
        # (cheap: 1 forward pass with hooks)
        # for simplicity skip detailed; report n_steps × ~0.2 as rough turnover

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

    # summary table
    print("\n" + "=" * 80)
    print("  SUMMARY — step size 효과")
    print("=" * 80)
    print(f"{'Scenario':<10} {'n_steps':>8} {'BSM CVaR':>10} {'NN CVaR':>10} "
          f"{'ratio':>7} {'+%':>7} {'wall':>7} {'BSM turn':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<10} {r['n_steps']:>8} {r['bsm_cvar']:>10.4f} "
              f"{r['buehler_cvar']:>10.4f} {r['ratio']:>7.3f} "
              f"{r['improvement_pct']:>+6.1f}% {r['wall_s']:>6.0f}s "
              f"{r['bsm_turnover']:>10.2f}")

    # save
    import json
    out_json = ROOT / "data" / "bench_step_size.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()
