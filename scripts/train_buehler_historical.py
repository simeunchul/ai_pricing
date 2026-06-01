"""B4 SPY historical hedging — train Buehler PG-on-CVaR on real-data bootstrap.

학습: SPY 일별 log-return 10년치 (block bootstrap, block_size=5)
평가: 같은 분포에서 fresh sampling
비교: BSM Δ (closed-form) baseline + GBM 학습 v1 모델
"""

from __future__ import annotations

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

from ai_hedging.agents.buehler_pg import HedgingPolicy
from ai_hedging.agents.buehler_pg_historical import (
    HistoricalConfig, fetch_spy_returns, annualized_vol,
    simulate_batch_historical, train_buehler_historical,
)


def bsm_delta_baseline_historical(cfg: HistoricalConfig, n_paths: int = 8000,
                                    device: str = "cpu", seed: int = 777) -> np.ndarray:
    """Run BSM Δ analytic on the SAME historical paths.

    Uses sigma from cfg (annualized realized vol) for delta computation.
    """
    import math
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    rets = cfg.historical_returns
    block = cfg.block_size
    n_steps = cfg.n_steps
    B = n_paths

    n_blocks = (n_steps + block - 1) // block
    starts = rng.integers(0, len(rets) - block, size=(B, n_blocks))
    sampled = np.empty((B, n_blocks * block), dtype=np.float32)
    for j in range(n_blocks):
        for k in range(block):
            sampled[:, j * block + k] = rets[starts[:, j] + k]
    log_rets = sampled[:, :n_steps]

    dt = cfg.T / cfg.n_steps
    exp_r_dt = math.exp(cfg.r * dt)

    from ai_hedging.agents.buehler_pg import _bsm_call_premium
    premium = _bsm_call_premium(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma)

    S = np.full(B, cfg.S0, dtype=np.float32)
    cash = np.full(B, premium, dtype=np.float32)
    hedge = np.zeros(B, dtype=np.float32)

    for t in range(n_steps):
        T_rem = max(cfg.T * (1 - t / n_steps), 1e-6)
        d1 = (np.log(S / cfg.K) + (cfg.r - cfg.q + 0.5 * cfg.sigma ** 2) * T_rem) \
             / (cfg.sigma * np.sqrt(T_rem))
        new_hedge = norm.cdf(d1).astype(np.float32)

        tc = cfg.tc_rate * np.abs(new_hedge - hedge) * S
        cash -= tc
        hedge = new_hedge

        S_next = S * np.exp(log_rets[:, t])
        cash = cash * exp_r_dt + hedge * (S_next - S)
        S = S_next

    payoff = np.maximum(S - cfg.K, 0).astype(np.float32)
    close_tc = cfg.tc_rate * np.abs(hedge) * S
    cash -= close_tc
    return cash - payoff * cfg.notional


def cvar_5(pnl: np.ndarray) -> float:
    return float(-np.sort(pnl)[:int(0.05 * len(pnl))].mean())


def main():
    print("=" * 75)
    print("  B4 SPY Historical — Buehler PG-on-CVaR (실데이터 환경)")
    print("=" * 75)

    print("\n[1] SPY 일별 log-return fetch ...")
    rets = fetch_spy_returns(start="2014-01-01", end="2026-04-26")
    sigma_realized = annualized_vol(rets)
    print(f"    n_returns={len(rets)}  realized annualized vol = {sigma_realized*100:.2f}%")
    print(f"    return stats: mean={rets.mean()*252*100:+.2f}%/yr  "
          f"std/d={rets.std()*100:.3f}%  "
          f"min/d={rets.min()*100:.2f}%  max/d={rets.max()*100:+.2f}%  "
          f"kurt={float((((rets-rets.mean())/rets.std())**4).mean()):.2f}")

    cfg = HistoricalConfig(
        S0=100.0, K=100.0, T=30 / 365, r=0.02, q=0.0,
        sigma=sigma_realized,
        n_steps=30, tc_rate=0.003,
        batch_size=4096, lr=1e-3, n_epochs=200,
        cvar_alpha=0.05, grad_clip=1.0, seed=0,
        action_low=0.0, action_high=1.0,
        block_size=5,
        historical_returns=rets,
    )

    out = str(ROOT / "models" / "buehler_historical.pt")
    print(f"\n[cfg] tc={cfg.tc_rate}  T={cfg.T:.4f}y  n_steps={cfg.n_steps}  "
          f"batch={cfg.batch_size}  epochs={cfg.n_epochs}  α={cfg.cvar_alpha}  "
          f"sigma_realized={cfg.sigma*100:.2f}%  block={cfg.block_size}")
    print()

    print("[2] Training Buehler-historical ...")
    t0 = time.time()
    res = train_buehler_historical(cfg, out_path=out, device="cpu",
                                    log_every=20, eval_batch=8000)
    elapsed = time.time() - t0

    pnl_buehler = res["eval_pnl"]
    cvar_buehler = res["final"]["cvar_5"]
    print(f"\n[Buehler-historical]  mean={pnl_buehler.mean():+.4f}  "
          f"std={pnl_buehler.std():.4f}  CVaR={cvar_buehler:.4f}  "
          f"wall={elapsed:.0f}s")

    print("\n[3] BSM Δ baseline on SAME historical paths ...")
    pnl_bsm = bsm_delta_baseline_historical(cfg, n_paths=8000, seed=777)
    cvar_bsm = cvar_5(pnl_bsm)
    print(f"[BSM Δ daily, historical paths]  mean={pnl_bsm.mean():+.4f}  "
          f"std={pnl_bsm.std():.4f}  CVaR={cvar_bsm:.4f}")

    ratio = cvar_buehler / cvar_bsm
    improvement = (1 - ratio) * 100
    if ratio < 0.80:
        verdict = "PASS (plan target ≥20%)"
    elif ratio < 1.0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print()
    print("=" * 75)
    print(f"  CVaR ratio (historical) = {ratio:.3f}    improvement = {improvement:+.1f}%")
    print(f"  Verdict (real-data env): {verdict}")
    print("=" * 75)

    # Compare with v1 GBM-trained model evaluated on historical paths
    print("\n[4] Cross-check: v1 GBM-trained model evaluated on HISTORICAL paths ...")
    v1_path = ROOT / "models" / "buehler_tc03_v2.pt"
    if v1_path.exists():
        ckpt = torch.load(v1_path, map_location="cpu", weights_only=False)
        v1_cfg = ckpt.get("cfg", {})
        v1_hidden = tuple(v1_cfg.get("hidden", cfg.hidden))
        v1_activation = v1_cfg.get("activation", cfg.activation)
        v1_policy = HedgingPolicy(
            hidden=v1_hidden, activation=v1_activation,
            action_low=cfg.action_low, action_high=cfg.action_high,
        )
        v1_policy.load_state_dict(ckpt["state_dict"])
        v1_policy.eval()
        with torch.no_grad():
            pnl_v1 = simulate_batch_historical(
                v1_policy, cfg, 8000, device="cpu", seed=cfg.seed + 9999
            ).cpu().numpy()
        cvar_v1 = cvar_5(pnl_v1)
        ratio_v1 = cvar_v1 / cvar_bsm
        print(f"    GBM-trained → historical eval: CVaR={cvar_v1:.4f}  "
              f"ratio={ratio_v1:.3f}  improvement={(1-ratio_v1)*100:+.1f}%")
        print(f"    (vs historical-trained: {improvement:+.1f}%)")
        cross_verdict = "잘 일반화됨" if ratio_v1 < 0.95 else "OOD 손실 큼"
        print(f"    OOD 진단: {cross_verdict}")
    else:
        print(f"    {v1_path} 없음 — cross-check 생략")
        cvar_v1 = None
        ratio_v1 = None

    log_path = ROOT / "data" / "bench_buehler_historical.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== B4 SPY Historical Buehler (TC={cfg.tc_rate}) ===\n")
        f.write(f"Wall: {elapsed:.0f}s, batch={cfg.batch_size}, "
                f"epochs={cfg.n_epochs}, alpha={cfg.cvar_alpha}\n")
        f.write(f"sigma_realized={cfg.sigma*100:.2f}%  block={cfg.block_size}\n\n")
        f.write(f"[Buehler-historical]  mean={pnl_buehler.mean():+.4f}  "
                f"std={pnl_buehler.std():.4f}  CVaR={cvar_buehler:.4f}\n")
        f.write(f"[BSM Δ daily]         mean={pnl_bsm.mean():+.4f}  "
                f"std={pnl_bsm.std():.4f}  CVaR={cvar_bsm:.4f}\n\n")
        f.write(f"ratio (historical) = {ratio:.3f}  improvement = {improvement:+.1f}%\n")
        f.write(f"verdict = {verdict}\n\n")
        if cvar_v1 is not None:
            f.write(f"[v1 GBM-trained on historical paths] CVaR={cvar_v1:.4f}  ratio={ratio_v1:.3f}\n\n")
        f.write("learning curve (every 20 ep):\n")
        h = res["history"]
        for i in range(0, len(h["epoch"]), 20):
            f.write(f"  ep {h['epoch'][i]:3d}: CVaR={h['cvar'][i]:+.4f} "
                    f"mean={h['mean_pnl'][i]:+.4f}  std={h['std_pnl'][i]:.4f}\n")
    print(f"\n→ {log_path}")


if __name__ == "__main__":
    main()
