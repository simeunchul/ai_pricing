"""Stress test — 기존 학습된 Buehler 정책을 jump 환경에서 평가.

핵심 질문: GBM-only 에서 학습한 정책이 비정형 (Merton jump) 환경에서도
BSM Δ 를 이기나? 즉 "예측 안 함" 철학이 환경이 험해져도 유지되나?

학습은 하지 않는다. 기존 models/buehler_tc03_v2.pt 를 그대로 쓰고
환경의 jump 파라미터만 키워가며 4 가지 시나리오에서 평가.

Outputs
-------
data/stress_test_buehler_jump.json
"""

from __future__ import annotations

import json
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

from ai_hedging.agents.buehler_pg import BuehlerConfig, HedgingPolicy, simulate_batch
from ai_hedging.env import HedgingEnv, HedgingEnvConfig
from ai_hedging.baselines.bsm_delta import BSMDeltaHedger


# --------------------------------------------------------------------------- #
# Scenarios — λ 와 jump 크기를 점진적으로 키움
# --------------------------------------------------------------------------- #
SCENARIOS = [
    dict(name="GBM-only (baseline)",
         jump_intensity=0.0,  jump_mean=0.0,    jump_std=0.0),
    dict(name="Small jump (일상 잡음)",
         jump_intensity=5.0,  jump_mean=-0.02,  jump_std=0.05),
    dict(name="Medium jump (분기 어닝쇼크)",
         jump_intensity=10.0, jump_mean=-0.05,  jump_std=0.10),
    dict(name="Large jump (코로나급 stress)",
         jump_intensity=20.0, jump_mean=-0.08,  jump_std=0.15),
]

MODEL_PATH = ROOT / "models" / "buehler_tc03_v2.pt"
N_PATHS = 8000
EVAL_SEED = 9999


def cvar_5(pnl: np.ndarray) -> float:
    n_tail = max(int(0.05 * len(pnl)), 1)
    return float(-np.sort(pnl)[:n_tail].mean())


def evaluate_buehler(cfg: BuehlerConfig, policy: HedgingPolicy,
                     n_paths: int, seed: int) -> dict:
    policy.eval()
    with torch.no_grad():
        pnl = simulate_batch(policy, cfg, n_paths, "cpu", seed=seed).cpu().numpy()
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "cvar_5": cvar_5(pnl),
        "p05": float(np.quantile(pnl, 0.05)),
        "pnl": pnl,
    }


def evaluate_bsm_delta(env_cfg: HedgingEnvConfig, n_paths: int, seed: int) -> dict:
    hedger = BSMDeltaHedger("call")
    finals = []
    for p in range(n_paths):
        env = HedgingEnv(env_cfg)
        env.reset(seed=seed + p)
        for _ in range(env_cfg.n_steps):
            h = hedger.act(env)
            obs, r, done, _, info = env.step(np.array([h], dtype=np.float32))
            if done:
                finals.append(info["terminal_pnl"])
                break
    arr = np.array(finals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "cvar_5": cvar_5(arr),
        "p05": float(np.quantile(arr, 0.05)),
        "pnl": arr,
    }


def main():
    print("=" * 78)
    print("  Buehler stress test — GBM-only 학습 정책을 jump 환경에서 평가")
    print(f"  Model: {MODEL_PATH.name}   N paths: {N_PATHS}")
    print("=" * 78)

    # Load policy ----------------------------------------------------------- #
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg_base = BuehlerConfig(**ckpt["cfg"])
    policy = HedgingPolicy(
        hidden=cfg_base.hidden, activation=cfg_base.activation,
        action_low=cfg_base.action_low, action_high=cfg_base.action_high,
    )
    policy.load_state_dict(ckpt["state_dict"])

    rows = []
    for sc in SCENARIOS:
        name = sc["name"]
        # Buehler config for batch sim (PyTorch)
        cfg_buehler = BuehlerConfig(**{
            **ckpt["cfg"],
            "jump_intensity": sc["jump_intensity"],
            "jump_mean":      sc["jump_mean"],
            "jump_std":       sc["jump_std"],
        })
        # Gym env for BSM Δ
        env_cfg = HedgingEnvConfig(
            S0=cfg_base.S0, K=cfg_base.K, T=cfg_base.T,
            r=cfg_base.r, q=cfg_base.q, sigma=cfg_base.sigma,
            n_steps=cfg_base.n_steps, tc_rate=cfg_base.tc_rate,
            opt=cfg_base.opt, reward_shaping=False,
            jump_intensity=sc["jump_intensity"],
            jump_mean=sc["jump_mean"],
            jump_std=sc["jump_std"],
        )

        t0 = time.time()
        b = evaluate_buehler(cfg_buehler, policy, N_PATHS, EVAL_SEED)
        bsm = evaluate_bsm_delta(env_cfg, n_paths=2000, seed=EVAL_SEED)
        elapsed = time.time() - t0

        ratio = b["cvar_5"] / bsm["cvar_5"] if bsm["cvar_5"] != 0 else float("nan")
        improvement_pct = (1 - ratio) * 100

        print()
        print(f"--- {name}")
        print(f"    λ={sc['jump_intensity']:.1f}/yr  μJ={sc['jump_mean']:+.3f}  σJ={sc['jump_std']:.3f}")
        print(f"    BSM Δ      CVaR@5%={bsm['cvar_5']:.4f}  mean={bsm['mean']:+.4f}  std={bsm['std']:.4f}")
        print(f"    Buehler    CVaR@5%={b  ['cvar_5']:.4f}  mean={b  ['mean']:+.4f}  std={b  ['std']:.4f}")
        print(f"    ratio Buehler/BSM = {ratio:.4f}  → {improvement_pct:+.2f}% improvement")
        print(f"    ({elapsed:.1f}s)")

        rows.append({
            "scenario":           name,
            "jump_intensity":     sc["jump_intensity"],
            "jump_mean":          sc["jump_mean"],
            "jump_std":           sc["jump_std"],
            "bsm_mean":           bsm["mean"],
            "bsm_std":            bsm["std"],
            "bsm_cvar_5":         bsm["cvar_5"],
            "bsm_p05":            bsm["p05"],
            "buehler_mean":       b["mean"],
            "buehler_std":        b["std"],
            "buehler_cvar_5":     b["cvar_5"],
            "buehler_p05":        b["p05"],
            "ratio":              ratio,
            "improvement_pct":    improvement_pct,
            "elapsed_s":          elapsed,
        })

    # Save JSON ------------------------------------------------------------- #
    out = {
        "model_path":   str(MODEL_PATH),
        "n_paths":      N_PATHS,
        "eval_seed":    EVAL_SEED,
        "model_cfg":    ckpt["cfg"],
        "scenarios":    rows,
    }
    out_path = ROOT / "data" / "stress_test_buehler_jump.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 78)
    print(f"  → {out_path}")
    print("=" * 78)
    print()
    print("  Summary table (Buehler/BSM CVaR ratio, lower=better):")
    for r in rows:
        bar = "█" * max(int((1 - r["ratio"]) * 40), 0) if r["ratio"] < 1 else ""
        sign = "✓" if r["ratio"] < 1 else "✗"
        print(f"    {sign} {r['scenario']:<32s}  ratio={r['ratio']:.3f}  {bar}")


if __name__ == "__main__":
    main()
