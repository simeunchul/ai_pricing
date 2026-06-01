"""헷지 전략 out-of-sample 검증 — BSM Δ vs Buehler PG vs no-hedge.

기존 buehler_historical.pt 는 2014~2026 전체 SPY 로 학습 → 평가 데이터 누설 의심.
이 script 는:
  1. SPY returns 를 train (2014~2023) / test (2024~2026) 로 분리
  2. Buehler PG 를 train data 로만 학습
  3. Test data 의 paths 에서 BSM Δ + Buehler + no-hedge 비교
  4. Stress windows (코로나 2020-03 등) 도 별도 평가
  5. 결과 JSON + console 표

한 줄로: "헷지 전략이 학습 안 본 시기에도 통하는가?"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_hedging" / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ai_hedging.agents.buehler_pg import (
    BuehlerConfig, HedgingPolicy, _bsm_call_premium, cvar_loss_fn,
)
from ai_hedging.agents.buehler_pg_historical import (
    HistoricalConfig, simulate_batch_historical, train_buehler_historical,
)
from pricing.bsm import BSMInputs
from pricing.greeks.analytic import call_greeks


# ────────────────────────────────────────────────────── data

def load_spy_returns_with_dates(cache_path: str = "data/macro_events/spy_returns.csv"
                                  ) -> tuple[np.ndarray, list[str]]:
    p = Path(cache_path)
    lines = p.read_text(encoding="utf-8").splitlines()[1:]
    dates = [line.split(",")[0] for line in lines]
    rets = np.array([float(line.split(",")[1]) for line in lines])
    return rets, dates


def split_train_test(rets: np.ndarray, dates: list[str], cutoff: str) -> tuple:
    """Train (date < cutoff) / Test (date >= cutoff). 반환 4-튜플."""
    cut_idx = next((i for i, d in enumerate(dates) if d >= cutoff), len(dates))
    return rets[:cut_idx], dates[:cut_idx], rets[cut_idx:], dates[cut_idx:]


# ────────────────────────────────────────────────────── BSM Δ baseline

def simulate_bsm_delta(
    rets_pool: np.ndarray, cfg: HistoricalConfig,
    n_paths: int = 4000, seed: int = 42,
) -> np.ndarray:
    """BSM Δ-hedger on historical bootstrap paths (pool 에서 sample).

    같은 seed 로 Buehler 와 동일 paths 사용 가능.
    """
    rng = np.random.default_rng(seed)
    n_steps = cfg.n_steps
    block = cfg.block_size
    n_blocks = (n_steps + block - 1) // block
    starts = rng.integers(0, len(rets_pool) - block, size=(n_paths, n_blocks))
    sampled = np.empty((n_paths, n_blocks * block), dtype=np.float64)
    for j in range(n_blocks):
        for k in range(block):
            sampled[:, j * block + k] = rets_pool[starts[:, j] + k]
    log_rets = sampled[:, :n_steps]

    dt = cfg.T / n_steps
    exp_r_dt = math.exp(cfg.r * dt)
    premium = _bsm_call_premium(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma)

    S = np.full(n_paths, cfg.S0)
    cash = np.full(n_paths, premium)
    hedge = np.zeros(n_paths)

    for t in range(n_steps):
        dt_rem = cfg.T * (1 - t / n_steps)
        # BSM Δ at each path's S
        delta = np.zeros(n_paths)
        for i in range(n_paths):
            inputs = BSMInputs(S[i], cfg.K, max(dt_rem, 1e-6), cfg.r, cfg.q, cfg.sigma)
            delta[i] = call_greeks(inputs).delta
        # 거래비용 + 리밸런싱
        tc = cfg.tc_rate * np.abs(delta - hedge) * S
        cash -= tc
        hedge = delta
        S_next = S * np.exp(log_rets[:, t])
        cash = cash * exp_r_dt + hedge * (S_next - S)
        S = S_next

    payoff = np.maximum(S - cfg.K, 0)
    close_tc = cfg.tc_rate * np.abs(hedge) * S
    cash -= close_tc
    return cash - payoff * cfg.notional


def simulate_no_hedge(
    rets_pool: np.ndarray, cfg: HistoricalConfig,
    n_paths: int = 4000, seed: int = 42,
) -> np.ndarray:
    """No-hedge: just collect premium and pay payoff. 헷지 효과 비교용 floor."""
    rng = np.random.default_rng(seed)
    n_steps = cfg.n_steps
    block = cfg.block_size
    n_blocks = (n_steps + block - 1) // block
    starts = rng.integers(0, len(rets_pool) - block, size=(n_paths, n_blocks))
    sampled = np.empty((n_paths, n_blocks * block), dtype=np.float64)
    for j in range(n_blocks):
        for k in range(block):
            sampled[:, j * block + k] = rets_pool[starts[:, j] + k]
    log_rets = sampled[:, :n_steps]

    dt = cfg.T / n_steps
    premium = _bsm_call_premium(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma)
    S = cfg.S0 * np.exp(log_rets.sum(axis=1))
    cash = premium * math.exp(cfg.r * cfg.T)
    payoff = np.maximum(S - cfg.K, 0)
    return cash - payoff * cfg.notional


# ────────────────────────────────────────────────────── Buehler eval

def simulate_buehler_eval(
    policy: HedgingPolicy, cfg: HistoricalConfig,
    rets_pool: np.ndarray, n_paths: int = 4000, seed: int = 42,
) -> np.ndarray:
    """Buehler policy eval on rets_pool with given seed."""
    cfg = HistoricalConfig(**{**cfg.__dict__, "historical_returns": rets_pool})
    pnl = simulate_batch_historical(
        policy, cfg, batch_size=n_paths, device="cpu", seed=seed,
    )
    return pnl.detach().cpu().numpy()


# ────────────────────────────────────────────────────── stats

def stats(pnl: np.ndarray, alpha: float = 0.05) -> dict:
    sorted_pnl = np.sort(pnl)
    n_tail = max(1, int(alpha * len(pnl)))
    cvar = -float(sorted_pnl[:n_tail].mean())
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "p05": float(np.quantile(pnl, 0.05)),
        "p95": float(np.quantile(pnl, 0.95)),
        "min": float(pnl.min()),
        "max": float(pnl.max()),
        "cvar_5": cvar,
        "n": len(pnl),
    }


def print_stats_table(label: str, s: dict):
    print(f"  {label:<28} mean={s['mean']:+.4f}  std={s['std']:.4f}  "
          f"CVaR_5={s['cvar_5']:.4f}  p05={s['p05']:+.4f}")


# ────────────────────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2024-01-01",
                    help="train < cutoff, test >= cutoff (default 2024-01-01)")
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--save", default=str(ROOT / "data" / "validate_hedging_oos.json"))
    ap.add_argument("--reuse-policy", action="store_true",
                    help="기존 buehler_historical.pt 재사용 (학습 skip)")
    args = ap.parse_args()

    print("=== Hedging OOS validation ===")
    rets, dates = load_spy_returns_with_dates()
    print(f"  total returns: {len(rets)}, span: {dates[0]} → {dates[-1]}")

    train_rets, train_dates, test_rets, test_dates = split_train_test(rets, dates, args.cutoff)
    print(f"  train: n={len(train_rets)}, {train_dates[0]} → {train_dates[-1]}")
    print(f"  test : n={len(test_rets)}, {test_dates[0]} → {test_dates[-1]}")
    print(f"  train ann.vol = {train_rets.std() * np.sqrt(252) * 100:.2f}%")
    print(f"  test  ann.vol = {test_rets.std() * np.sqrt(252) * 100:.2f}%")

    # config: 30일 ATM call, TC=30bps
    base_cfg = HistoricalConfig(
        S0=100.0, K=100.0, T=30/365, r=0.02, sigma=float(train_rets.std() * np.sqrt(252)),
        n_steps=30, tc_rate=0.003, batch_size=2048,
        n_epochs=args.epochs, cvar_alpha=0.05,
        historical_returns=train_rets,
    )
    print(f"\n  cfg: T=30d, TC=30bps, sigma_train={base_cfg.sigma*100:.2f}%, "
          f"n_steps={base_cfg.n_steps}, n_epochs={base_cfg.n_epochs}")

    # ── 1. Train Buehler on TRAIN only
    out = {}
    if args.reuse_policy and Path("models/buehler_historical.pt").exists():
        print("\n[1] Reusing existing buehler_historical.pt (warning: trained on full data)")
        ckpt = torch.load("models/buehler_historical.pt", weights_only=False)
        policy = HedgingPolicy(
            hidden=base_cfg.hidden, activation=base_cfg.activation,
            action_low=base_cfg.action_low, action_high=base_cfg.action_high,
        )
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
    else:
        print(f"\n[1] Training Buehler on TRAIN data ({args.epochs} epochs)...")
        t0 = time.time()
        train_result = train_buehler_historical(
            base_cfg,
            out_path=str(ROOT / "models" / "buehler_oos_train.pt"),
            log_every=20,
        )
        print(f"  train wall: {time.time() - t0:.0f}s")
        # Reload trained policy
        ckpt = torch.load(ROOT / "models" / "buehler_oos_train.pt", weights_only=False)
        policy = HedgingPolicy(
            hidden=base_cfg.hidden, activation=base_cfg.activation,
            action_low=base_cfg.action_low, action_high=base_cfg.action_high,
        )
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        out["train_history_final"] = train_result["final"]

    # ── 2. Evaluate all 3 strategies on TEST paths
    print(f"\n[2] Evaluate on TEST paths (n={args.n_paths}, OOS)...")
    seed = 9999
    no_pnl = simulate_no_hedge(test_rets, base_cfg, args.n_paths, seed)
    bsm_pnl = simulate_bsm_delta(test_rets, base_cfg, args.n_paths, seed)
    buehler_pnl = simulate_buehler_eval(policy, base_cfg, test_rets, args.n_paths, seed)

    print(f"\n  === TEST set results (n={args.n_paths}, OOS) ===")
    s_no = stats(no_pnl); print_stats_table("NO HEDGE", s_no)
    s_bsm = stats(bsm_pnl); print_stats_table("BSM Δ daily", s_bsm)
    s_bue = stats(buehler_pnl); print_stats_table("Buehler PG (OOS)", s_bue)

    # ── 3. CVaR / mean lift vs no-hedge
    print(f"\n  === Lift vs NO HEDGE ===")
    print(f"  BSM Δ      CVaR lift = {(s_no['cvar_5'] - s_bsm['cvar_5']) / s_no['cvar_5'] * 100:+.1f}%   "
          f"std lift = {(s_no['std'] - s_bsm['std']) / s_no['std'] * 100:+.1f}%")
    print(f"  Buehler    CVaR lift = {(s_no['cvar_5'] - s_bue['cvar_5']) / s_no['cvar_5'] * 100:+.1f}%   "
          f"std lift = {(s_no['std'] - s_bue['std']) / s_no['std'] * 100:+.1f}%")

    print(f"\n  === Buehler vs BSM ===")
    print(f"  CVaR diff  = {s_bsm['cvar_5'] - s_bue['cvar_5']:+.4f}  "
          f"({(s_bsm['cvar_5'] - s_bue['cvar_5']) / s_bsm['cvar_5'] * 100:+.1f}% improvement)")
    print(f"  mean diff  = {s_bue['mean'] - s_bsm['mean']:+.4f}")
    print(f"  std diff   = {s_bue['std'] - s_bsm['std']:+.4f}  "
          f"({(s_bsm['std'] - s_bue['std']) / s_bsm['std'] * 100:+.1f}% reduction)")

    # ── 4. Stress windows
    print(f"\n[3] Stress regime tests (2020 COVID, 2022 rate-hike, 2024 recent)...")
    stress_windows = [
        ("2020-COVID",     "2020-02-15", "2020-04-30"),
        ("2022-rate-hike", "2022-01-01", "2022-12-31"),
        ("2024-recent",    "2024-01-01", "2024-12-31"),
    ]
    stress_results = []
    for name, start, end in stress_windows:
        mask = [(d >= start and d <= end) for d in dates]
        sub_rets = rets[np.array(mask)]
        if len(sub_rets) < 60:
            print(f"  [{name}] skip (only {len(sub_rets)} days)")
            continue
        print(f"  [{name}] n={len(sub_rets)} days, vol={sub_rets.std()*np.sqrt(252)*100:.1f}%")
        # 같은 paths 로 둘 다
        s_no_w = stats(simulate_no_hedge(sub_rets, base_cfg, 2000, 1111))
        s_bsm_w = stats(simulate_bsm_delta(sub_rets, base_cfg, 2000, 1111))
        s_bue_w = stats(simulate_buehler_eval(policy, base_cfg, sub_rets, 2000, 1111))
        print(f"    NO HEDGE     CVaR={s_no_w['cvar_5']:.3f}  std={s_no_w['std']:.3f}")
        print(f"    BSM Δ        CVaR={s_bsm_w['cvar_5']:.3f}  std={s_bsm_w['std']:.3f}  "
              f"(vs NO: CVaR {(s_no_w['cvar_5']-s_bsm_w['cvar_5'])/s_no_w['cvar_5']*100:+.0f}%)")
        print(f"    Buehler      CVaR={s_bue_w['cvar_5']:.3f}  std={s_bue_w['std']:.3f}  "
              f"(vs BSM: CVaR {(s_bsm_w['cvar_5']-s_bue_w['cvar_5'])/s_bsm_w['cvar_5']*100:+.0f}%)")
        stress_results.append({
            "name": name, "start": start, "end": end, "n_days": len(sub_rets),
            "vol": float(sub_rets.std() * np.sqrt(252)),
            "no_hedge": s_no_w, "bsm": s_bsm_w, "buehler": s_bue_w,
        })

    # ── 5. Save
    out["test"] = {
        "no_hedge": s_no, "bsm": s_bsm, "buehler": s_bue,
        "n_paths": args.n_paths,
    }
    out["train_dates"] = (train_dates[0], train_dates[-1], len(train_rets))
    out["test_dates"] = (test_dates[0], test_dates[-1], len(test_rets))
    out["stress"] = stress_results
    out["cfg"] = {"T_days": 30, "TC_bps": 30, "sigma_train": base_cfg.sigma}
    Path(args.save).write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8",
    )
    print(f"\n→ saved: {args.save}")


if __name__ == "__main__":
    main()
