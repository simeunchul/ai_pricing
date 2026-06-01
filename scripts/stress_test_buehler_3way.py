"""3-way stress test: GBM학습 vs jump-fixed 학습 vs jump-random 학습.

세 정책을 동일한 4 시나리오 (GBM-only / Small / Medium / Large jump) 에서
BSM Δ baseline 대비 평가하고 한 번에 비교.

세 정책 모두:
  - 정책 입력 5-dim 동일 (jump 신호 0개)
  - 가중치/구조/하이퍼파라미터 동일 (학습 분포만 차이)

Outputs
-------
data/stress_test_buehler_3way.json
"""

from __future__ import annotations

import json
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


SCENARIOS = [
    dict(name="GBM-only",      jump_intensity=0.0,  jump_mean=0.0,    jump_std=0.0),
    dict(name="Small jump",    jump_intensity=5.0,  jump_mean=-0.02,  jump_std=0.05),
    dict(name="Medium jump",   jump_intensity=10.0, jump_mean=-0.05,  jump_std=0.10),
    dict(name="Large jump",    jump_intensity=20.0, jump_mean=-0.08,  jump_std=0.15),
]

MODELS = [
    dict(label="GBM학습 (기존)",          path=ROOT / "models" / "buehler_tc03_v2.pt"),
    dict(label="Jump-fixed 학습 (λ=10)",  path=ROOT / "models" / "buehler_jump_fixed.pt"),
    dict(label="Jump-random 학습",        path=ROOT / "models" / "buehler_jump_random.pt"),
]

N_PATHS = 8000
EVAL_SEED = 9999


def cvar_5(pnl: np.ndarray) -> float:
    n_tail = max(int(0.05 * len(pnl)), 1)
    return float(-np.sort(pnl)[:n_tail].mean())


def load_policy(path: Path) -> tuple[HedgingPolicy, BuehlerConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = {k: v for k, v in ckpt["cfg"].items() if not k.startswith("_")}
    cfg = BuehlerConfig(**cfg_dict)
    pol = HedgingPolicy(
        hidden=cfg.hidden, activation=cfg.activation,
        action_low=cfg.action_low, action_high=cfg.action_high,
    )
    pol.load_state_dict(ckpt["state_dict"])
    pol.eval()
    return pol, cfg


def eval_buehler(policy: HedgingPolicy, base_cfg: BuehlerConfig, sc: dict,
                 n_paths: int, seed: int) -> dict:
    cfg_eval = BuehlerConfig(**{
        **base_cfg.__dict__,
        "jump_intensity": sc["jump_intensity"],
        "jump_mean":      sc["jump_mean"],
        "jump_std":       sc["jump_std"],
    })
    with torch.no_grad():
        pnl = simulate_batch(policy, cfg_eval, n_paths, "cpu", seed=seed).cpu().numpy()
    return {"mean": float(pnl.mean()), "std": float(pnl.std()),
            "cvar_5": cvar_5(pnl), "p05": float(np.quantile(pnl, 0.05))}


def eval_bsm(base_cfg: BuehlerConfig, sc: dict, n_paths: int, seed: int) -> dict:
    env_cfg = HedgingEnvConfig(
        S0=base_cfg.S0, K=base_cfg.K, T=base_cfg.T,
        r=base_cfg.r, q=base_cfg.q, sigma=base_cfg.sigma,
        n_steps=base_cfg.n_steps, tc_rate=base_cfg.tc_rate,
        opt=base_cfg.opt, reward_shaping=False,
        jump_intensity=sc["jump_intensity"],
        jump_mean=sc["jump_mean"],
        jump_std=sc["jump_std"],
    )
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
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "cvar_5": cvar_5(arr), "p05": float(np.quantile(arr, 0.05))}


def main():
    print("=" * 90)
    print("  3-way Buehler stress test  ·  3 모델 × 4 시나리오 vs BSM Δ baseline")
    print(f"  N paths: {N_PATHS}")
    print("=" * 90)

    # Load 3 policies
    policies = []
    for m in MODELS:
        pol, cfg = load_policy(m["path"])
        policies.append({"label": m["label"], "policy": pol, "cfg": cfg, "path": str(m["path"])})
        print(f"  loaded: {m['label']:<32s}  ({m['path'].name})")
    print()

    # BSM baseline first (independent of policy)
    bsm_by_scenario = {}
    print(">> BSM Δ baseline ...")
    for sc in SCENARIOS:
        t0 = time.time()
        bsm = eval_bsm(policies[0]["cfg"], sc, n_paths=2000, seed=EVAL_SEED)
        elapsed = time.time() - t0
        bsm_by_scenario[sc["name"]] = bsm
        print(f"   {sc['name']:<14s}  CVaR={bsm['cvar_5']:7.4f}  ({elapsed:.1f}s)")
    print()

    # Each policy in each scenario
    rows = []
    for pol_info in policies:
        print(f">> {pol_info['label']}")
        for sc in SCENARIOS:
            t0 = time.time()
            b   = eval_buehler(pol_info["policy"], pol_info["cfg"], sc, N_PATHS, EVAL_SEED)
            bsm = bsm_by_scenario[sc["name"]]
            ratio = b["cvar_5"] / bsm["cvar_5"] if bsm["cvar_5"] != 0 else float("nan")
            improvement = (1 - ratio) * 100
            elapsed = time.time() - t0
            print(f"   {sc['name']:<14s}  Buehler CVaR={b['cvar_5']:7.4f}  "
                  f"ratio={ratio:.3f}  improvement={improvement:+6.2f}%  ({elapsed:.1f}s)")
            rows.append({
                "model_label":     pol_info["label"],
                "model_path":      pol_info["path"],
                "scenario":        sc["name"],
                "jump_intensity":  sc["jump_intensity"],
                "jump_mean":       sc["jump_mean"],
                "jump_std":        sc["jump_std"],
                "buehler_mean":    b["mean"], "buehler_std": b["std"],
                "buehler_cvar_5":  b["cvar_5"], "buehler_p05": b["p05"],
                "bsm_mean":        bsm["mean"], "bsm_std": bsm["std"],
                "bsm_cvar_5":      bsm["cvar_5"], "bsm_p05": bsm["p05"],
                "ratio":           ratio,
                "improvement_pct": improvement,
            })
        print()

    # Summary matrix
    print("=" * 90)
    print("  Summary — improvement % by (model × scenario), positive = Buehler 승")
    print("=" * 90)
    header = f"  {'model':<32s} | " + " | ".join(f"{sc['name']:>14s}" for sc in SCENARIOS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pol_info in policies:
        cells = []
        for sc in SCENARIOS:
            row = next(r for r in rows
                       if r["model_label"] == pol_info["label"] and r["scenario"] == sc["name"])
            sign = "+" if row["improvement_pct"] >= 0 else ""
            cells.append(f"{sign}{row['improvement_pct']:>13.2f}%")
        print(f"  {pol_info['label']:<32s} | " + " | ".join(cells))
    print()

    # Save JSON
    out_path = ROOT / "data" / "stress_test_buehler_3way.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_paths": N_PATHS,
            "eval_seed": EVAL_SEED,
            "scenarios": SCENARIOS,
            "models": [{"label": m["label"], "path": str(m["path"])} for m in MODELS],
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"  → {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
