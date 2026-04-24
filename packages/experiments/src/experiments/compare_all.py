"""End-of-Week-7 integration: compare all 5 pricing methods and hedging strategies.

Produces:
  - `data/all_methods_pricing.csv` : per-method pricing stats
  - `data/all_methods_hedging.csv` : per-method hedging stats (PnL distribution metrics)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pricing.bsm import BSMInputs, call_price

from ai_hedging.baselines.bsm_delta import BSMDeltaHedger, rebalance_interval
from ai_hedging.env import HedgingEnvConfig

from experiments.metrics import (
    pricing_stats_from_arrays,
    hedging_stats_from_arrays,
)
from experiments.compare_pricers import _bsm_panel, bench_bsm, bench_nn_pricer


def run_all(
    out_dir: str = "data",
    nn_model: str = "models/nn_pricer.pt",
    ppo_model: str = "models/ppo_hedger.zip",
    tc_rate: float = 0.003,
    n_panel: int = 2000,
    n_hedge_paths: int = 1000,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Pricing comparison ---------------------------------------
    raw, truth = _bsm_panel(n_panel)
    rows_price = []

    bsm_pred, ms_bsm = bench_bsm(raw)
    rows_price.append(pricing_stats_from_arrays("A-BSM", bsm_pred, truth, ms_bsm).as_row())

    if Path(nn_model).exists():
        nn_pred, ms_nn = bench_nn_pricer(raw, nn_model)
        if nn_pred is not None:
            rows_price.append(pricing_stats_from_arrays("B1-NN", nn_pred, truth, ms_nn).as_row())

    df_price = pd.DataFrame(rows_price)
    df_price.to_csv(out / "all_methods_pricing.csv", index=False)
    print("\n=== Pricing comparison ===")
    print(df_price.to_string(index=False))

    # ---------- Hedging comparison ---------------------------------------
    cfg = HedgingEnvConfig(tc_rate=tc_rate)
    rows_hedge = []

    # BSM delta with various rebalance frequencies
    hedger = BSMDeltaHedger(opt="call")
    for every, name in [(1, "A-BSMΔ-daily"), (5, "A-BSMΔ-weekly")]:
        t0 = time.perf_counter()
        res = rebalance_interval(cfg, every=every, hedger=hedger, n_paths=n_hedge_paths)
        ms = 1000 * (time.perf_counter() - t0) / n_hedge_paths
        turn = float(res.get("pnl_array", np.array([])).std())  # proxy
        stat = hedging_stats_from_arrays(name, res["pnl_array"], turn, ms)
        rows_hedge.append(stat.as_row())

    # PPO agent
    if Path(ppo_model).exists():
        try:
            from ai_hedging.agents.ppo_hedger import evaluate_ppo
            t0 = time.perf_counter()
            res = evaluate_ppo(ppo_model, cfg, n_paths=n_hedge_paths)
            ms = 1000 * (time.perf_counter() - t0) / n_hedge_paths
            stat = hedging_stats_from_arrays("B4-DeepHedging-PPO", res["pnl_array"], 0.0, ms)
            rows_hedge.append(stat.as_row())
        except ImportError as e:
            print(f"[compare_all] skip PPO eval: {e}")

    df_hedge = pd.DataFrame(rows_hedge)
    df_hedge.to_csv(out / "all_methods_hedging.csv", index=False)
    print("\n=== Hedging comparison ===")
    print(df_hedge.to_string(index=False))

    return {"pricing": df_price, "hedging": df_hedge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--nn-model", default="models/nn_pricer.pt")
    ap.add_argument("--ppo-model", default="models/ppo_hedger.zip")
    ap.add_argument("--tc", type=float, default=0.003)
    ap.add_argument("--n-panel", type=int, default=2000)
    ap.add_argument("--n-hedge", type=int, default=1000)
    args = ap.parse_args()
    run_all(args.out, args.nn_model, args.ppo_model, args.tc, args.n_panel, args.n_hedge)


if __name__ == "__main__":
    main()
