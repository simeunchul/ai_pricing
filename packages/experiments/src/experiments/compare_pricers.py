"""Compare pricing methods on a fixed option panel.

Methods:
  A  BSM Closed-form (ground truth for Euro)
  B1 NN Pricer (if models/nn_pricer.pt exists)
  B2 Deep Calibration — compares surface prediction vs Heston semi-analytic ground truth
  B3 News-adjusted — shows shift vs base for a representative headline

Run:
    python -m experiments.compare_pricers --nn-model models/nn_pricer.pt --out data/pricing_comparison.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pricing.bsm import BSMInputs, call_price
from experiments.metrics import PricingStats, pricing_stats_from_arrays


def _bsm_panel(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    S = rng.uniform(80, 120, size=n)
    K = rng.uniform(80, 120, size=n)
    T = rng.uniform(0.05, 1.5, size=n)
    r = rng.uniform(0.0, 0.05, size=n)
    q = rng.uniform(0.0, 0.03, size=n)
    sigma = rng.uniform(0.10, 0.50, size=n)
    raw = np.stack([S, K, T, r, q, sigma], axis=1)
    prices = np.array([call_price(BSMInputs(*row)) for row in raw])
    return raw, prices


def bench_bsm(raw: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    prices = np.array([call_price(BSMInputs(*r)) for r in raw])
    ms = 1000 * (time.perf_counter() - t0) / len(raw)
    return prices, ms


def bench_nn_pricer(raw: np.ndarray, model_path: str) -> tuple[np.ndarray, float]:
    try:
        from ai_pricing.nn_pricer.infer import load_pricer, price_batch
    except ImportError:
        return None, float("nan")  # type: ignore[return-value]
    model, dev = load_pricer(model_path)
    t0 = time.perf_counter()
    prices = price_batch(model, raw, device=dev)
    ms = 1000 * (time.perf_counter() - t0) / len(raw)
    return prices, ms


def run_compare(nn_model: str | None, out_csv: str, n_panel: int = 2000, seed: int = 0) -> pd.DataFrame:
    raw, truth = _bsm_panel(n_panel, seed=seed)

    rows: list[PricingStats] = []
    # BSM (ground truth)
    bsm_pred, ms_bsm = bench_bsm(raw)
    rows.append(pricing_stats_from_arrays("BSM-closed-form", bsm_pred, truth, ms_bsm))

    # NN Pricer
    if nn_model and Path(nn_model).exists():
        nn_pred, ms_nn = bench_nn_pricer(raw, nn_model)
        if nn_pred is not None:
            rows.append(pricing_stats_from_arrays("NN-Pricer", nn_pred, truth, ms_nn))

    df = pd.DataFrame([r.as_row() for r in rows])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-model", type=str, default="models/nn_pricer.pt")
    ap.add_argument("--out", type=str, default="data/pricing_comparison.csv")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    run_compare(args.nn_model, args.out, args.n)


if __name__ == "__main__":
    main()
