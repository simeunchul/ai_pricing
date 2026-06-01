"""SPY 실데이터 + 10분 step 학습 — 진짜 best 후보 검증.

이전 결과:
  - SPY 실데이터, daily, TC=30bps     : +31.3%
  - 합성 Jump,    10min, TC=10bps     : +29.1%
  - SPY 실데이터, 10min, TC=10bps     : ?  ← 이번 실험

가설: 실데이터 fat-tail × 짧은 step 시너지로 +35~40% 가능

데이터:
  - yfinance SPY 10m interval, 60일치 (≈ 2,340 봉)
  - block bootstrap (block_size=5, 50분 블록)
  - n_steps=4320 (30일 × 24h × 6 = 4320 of 10min) — wrap 가능

학습:
  - batch=1024, epoch=100 (CPU 1-2시간 예상)
"""

from __future__ import annotations

import math
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

from ai_hedging.agents.buehler_pg import HedgingPolicy, _bsm_call_premium, cvar_loss_fn
from ai_hedging.agents.buehler_pg_historical import (
    HistoricalConfig, simulate_batch_historical, train_buehler_historical,
)


def fetch_spy_15min(cache_path: str = "data/macro_events/spy_15min.csv") -> np.ndarray:
    """SPY 15분봉 fetch (yfinance, 60일치 무료. 10m 미지원으로 15m 사용)."""
    import time as time_mod
    cache = Path(cache_path)
    if cache.exists() and (time_mod.time() - cache.stat().st_mtime) < 3 * 86400:
        print(f"[spy 15m] cache hit: {cache}")
        lines = cache.read_text(encoding="utf-8").splitlines()[1:]
        return np.array([float(line.split(",")[1]) for line in lines])

    import yfinance as yf
    print(f"[spy 15m] fetching SPY 15min bars (60d) ...")
    df = yf.Ticker("SPY").history(period="60d", interval="15m", auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned empty for SPY 15m")
    closes = df["Close"].values
    log_rets = np.diff(np.log(closes))

    cache.parent.mkdir(parents=True, exist_ok=True)
    times = [t.strftime("%Y-%m-%d %H:%M:%S") for t in df.index[1:]]
    cache.write_text(
        "datetime,log_return\n" + "\n".join(f"{t},{r}" for t, r in zip(times, log_rets)),
        encoding="utf-8",
    )
    print(f"[spy 15m] {len(log_rets)} 15min log-returns saved → {cache}")
    return log_rets


def bsm_delta_baseline_historical_intraday(cfg: HistoricalConfig, n_paths: int,
                                            device="cpu", seed=777):
    """BSM Δ on SAME historical bootstrap paths (10min step)."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    rets = cfg.historical_returns
    block = cfg.block_size
    n_steps = cfg.n_steps
    B = n_paths

    n_blocks = (n_steps + block - 1) // block
    if len(rets) <= block:
        raise RuntimeError(f"data too small: {len(rets)} < block {block}")
    starts = rng.integers(0, len(rets) - block, size=(B, n_blocks))
    sampled = np.empty((B, n_blocks * block), dtype=np.float32)
    for j in range(n_blocks):
        for k in range(block):
            sampled[:, j * block + k] = rets[starts[:, j] + k]
    log_rets = sampled[:, :n_steps]

    dt = cfg.T / cfg.n_steps
    exp_r_dt = math.exp(cfg.r * dt)
    premium = _bsm_call_premium(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma)

    S = np.full(B, cfg.S0, dtype=np.float32)
    cash = np.full(B, premium, dtype=np.float32)
    hedge = np.zeros(B, dtype=np.float32)
    turnover = np.zeros(B, dtype=np.float32)

    for t in range(n_steps):
        T_rem = max(cfg.T * (1 - t / n_steps), 1e-6)
        d1 = (np.log(S / cfg.K) + (cfg.r - cfg.q + 0.5 * cfg.sigma ** 2) * T_rem) \
             / (cfg.sigma * np.sqrt(T_rem))
        new_hedge = norm.cdf(d1).astype(np.float32)

        delta_h = np.abs(new_hedge - hedge)
        tc = cfg.tc_rate * delta_h * S
        cash -= tc
        turnover += delta_h
        hedge = new_hedge

        S_next = S * np.exp(log_rets[:, t])
        cash = cash * exp_r_dt + hedge * (S_next - S)
        S = S_next

    payoff = np.maximum(S - cfg.K, 0).astype(np.float32)
    close_tc = cfg.tc_rate * np.abs(hedge) * S
    cash -= close_tc
    pnl = cash - payoff * cfg.notional
    return {
        "pnl": pnl,
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "cvar_5": float(-np.sort(pnl)[:int(0.05 * len(pnl))].mean()),
        "turnover": float(turnover.mean()),
    }


def main():
    print("=" * 80)
    print("  SPY 실데이터 + 10분 step — 진짜 best 후보")
    print("=" * 80)

    print("\n[1] SPY 15분봉 fetch ... (yfinance 10m 미지원으로 15m 사용)")
    rets = fetch_spy_15min()
    # annualized vol: 15min bars, 252 trading days × 6.5h × 4 (15min) = 6552 bars/year
    bars_per_year = 252 * 6.5 * 4
    sigma_realized = float(rets.std() * math.sqrt(bars_per_year))
    print(f"    n_returns={len(rets)}  realized annualized vol = {sigma_realized*100:.2f}%")
    print(f"    return stats: std/bar={rets.std()*100:.4f}%  "
          f"min={rets.min()*100:+.3f}%  max={rets.max()*100:+.3f}%  "
          f"kurt={float((((rets-rets.mean())/rets.std())**4).mean()):.2f}")

    cfg = HistoricalConfig(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0,
        sigma=sigma_realized,
        n_steps=2880,            # 15분 × 30일 × 24h × 4 = 2880 (calendar 기준)
        tc_rate=0.0010,          # 10bps (intraday 가정 — 일관성)
        notional=1.0, opt="call",
        hidden=(128, 128, 64), activation="relu",
        batch_size=1024,
        lr=1e-3, n_epochs=100,
        cvar_alpha=0.05, grad_clip=1.0, seed=0,
        action_low=0.0, action_high=1.0,
        block_size=5,            # 50분 블록 (vol clustering)
        historical_returns=rets,
    )

    print(f"\n[cfg] n_steps={cfg.n_steps}  batch={cfg.batch_size}  epochs={cfg.n_epochs}")
    print(f"      tc={cfg.tc_rate*1e4:.0f}bps  σ_realized={cfg.sigma*100:.2f}%  block={cfg.block_size}")
    print()

    out_path = str(ROOT / "models" / "buehler_spy_15min.pt")
    print("[2] Training Buehler — SPY 15min historical bootstrap ...")
    t0 = time.time()
    res = train_buehler_historical(cfg, out_path=out_path, device="cpu",
                                    log_every=10, eval_batch=8000)
    elapsed = time.time() - t0

    cvar_b = res["final"]["cvar_5"]
    mean_b = res["final"]["mean_pnl"]
    std_b = res["final"]["std_pnl"]
    print(f"\n[Buehler SPY-15min] CVaR={cvar_b:.4f}  mean={mean_b:+.4f}  "
          f"std={std_b:.4f}  wall={elapsed:.0f}s")

    print("\n[3] BSM Δ baseline on SAME historical paths ...")
    bsm = bsm_delta_baseline_historical_intraday(cfg, n_paths=8000, seed=777)
    cvar_bsm = bsm["cvar_5"]
    print(f"[BSM Δ 15min historical] CVaR={cvar_bsm:.4f}  mean={bsm['mean']:+.4f}  "
          f"std={bsm['std']:.4f}  turnover={bsm['turnover']:.2f}")

    ratio = cvar_b / cvar_bsm
    improvement = (1 - ratio) * 100
    if ratio < 0.80:
        verdict = "PASS (≥20%)"
    elif ratio < 1.0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    print()
    print("=" * 80)
    print(f"  RESULT — SPY 실데이터 + 15min step (yfinance 10m 미지원)")
    print(f"  Buehler CVaR = {cvar_b:.4f}   BSM CVaR = {cvar_bsm:.4f}")
    print(f"  ratio = {ratio:.3f}   improvement = {improvement:+.1f}%   verdict: {verdict}")
    print("=" * 80)

    print("\n[비교] 모든 시나리오 통합:")
    print(f"  GBM,    daily,  TC=30bps, 600ep : +24.5%")
    print(f"  SPY 실, daily,  TC=30bps, 200ep : +31.3%")
    print(f"  Jump,   10min,  TC=10bps, 100ep : +29.1%")
    print(f"  SPY 실, 15min,  TC=10bps, 100ep : {improvement:+.1f}%  ← 이번")

    import json
    out_json = ROOT / "data" / "bench_spy_10min.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({
            "n_returns": len(rets),
            "sigma_realized": sigma_realized,
            "kurt": float((((rets-rets.mean())/rets.std())**4).mean()),
            "n_steps": cfg.n_steps,
            "epochs": cfg.n_epochs,
            "batch": cfg.batch_size,
            "tc_rate": cfg.tc_rate,
            "wall_s": elapsed,
            "buehler_cvar": cvar_b, "buehler_mean": mean_b, "buehler_std": std_b,
            "bsm_cvar": cvar_bsm, "bsm_mean": bsm["mean"], "bsm_turnover": bsm["turnover"],
            "ratio": ratio, "improvement_pct": improvement, "verdict": verdict,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()
