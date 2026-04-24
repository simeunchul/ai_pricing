"""Rigorous evaluation of trained NN Pricer.

Implements the 3 standards from docs/nn_pricer_underfit_analysis.html:
  1. IV-space error  (업계 표준, 모든 moneyness 에서 의미 있는 단위)
  2. Moneyness-stratified report (deep OTM / OTM / ATM / ITM / deep ITM)
  3. Intrinsic-filter (intrinsic 바로 위 샘플 제외)

Usage:
    python scripts/bench_nn_pricer.py
    python scripts/bench_nn_pricer.py --model models/nn_pricer_rel.pt --n 10000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("packages/pricing/src")))
sys.path.insert(0, str(Path("packages/ai_pricing/src")))

from pricing.bsm import BSMInputs, call_price  # noqa: E402
from pricing.iv import implied_vol  # noqa: E402
from ai_pricing.nn_pricer.data import sample_inputs, label_bsm  # noqa: E402
from ai_pricing.nn_pricer.infer import load_pricer, price_batch  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def intrinsic_call(S, K, T, r):
    """Discounted intrinsic value for European call."""
    return np.maximum(S - K * np.exp(-r * T), 0.0)


def iv_batch(prices: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Vectorized IV solver. Returns NaN where solver fails."""
    ivs = np.full(len(prices), np.nan)
    for i, (S, K, T, r, q, _sigma) in enumerate(X):
        try:
            p = float(prices[i])
            # Price must be in (intrinsic, S) for IV to exist
            lo = intrinsic_call(S, K, T, r)
            if p < lo + 1e-9 or p > S - 1e-9:
                continue
            ivs[i] = implied_vol(p, BSMInputs(S, K, T, r, q, 0.2), opt="call")
        except (ValueError, ZeroDivisionError):
            pass
    return ivs


def moneyness_buckets(X: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return list of (name, bool_mask) for 5 standard buckets."""
    S, K = X[:, 0], X[:, 1]
    m = S / K
    return [
        ("Deep OTM",  m < 0.90),
        ("OTM",       (m >= 0.90) & (m < 0.97)),
        ("ATM",       (m >= 0.97) & (m <= 1.03)),
        ("ITM",       (m > 1.03) & (m <= 1.10)),
        ("Deep ITM",  m > 1.10),
    ]


def _fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def _fmt_vol_pts(x: float) -> str:
    """Vol point = 0.01. IV diff of 0.005 → '0.5 vol pts'."""
    return f"{x*100:6.3f} vp"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run(model_path: str, n: int, seed: int = 999, label: str = ""):
    print("=" * 78)
    title = f"NN Pricer benchmark  (n={n}, model={model_path})"
    if label:
        title = f"[{label}] " + title
    print(title)
    print("=" * 78)

    X = sample_inputs(n, seed=seed)
    y_true = label_bsm(X)

    # ---- NN inference (CPU) --------------------------------------------------
    model, dev = load_pricer(model_path, device="cpu")
    t0 = time.perf_counter()
    y_nn = price_batch(model, X, device=dev)
    t_nn = time.perf_counter() - t0

    # ---- BSM reference timing ------------------------------------------------
    t0 = time.perf_counter()
    _ = np.array([call_price(BSMInputs(*row)) for row in X])
    t_bsm = time.perf_counter() - t0

    # =========================================================================
    # A. Price-space raw metrics (naive, for comparison)
    # =========================================================================
    abs_err = np.abs(y_nn - y_true)
    rel_err_naive = abs_err / np.maximum(y_true, 1e-4)
    print(f"\n[A] Price-space (naive, w/ eps=1e-4 floor)")
    print(f"    MSE               : {(abs_err**2).mean():.2e}")
    print(f"    Mean rel err      : {rel_err_naive.mean()*100:7.2f}%")
    print(f"    P95 rel err       : {np.quantile(rel_err_naive, 0.95)*100:7.2f}%")

    # =========================================================================
    # B. Intrinsic-filtered price-space
    # =========================================================================
    S, K, T, r = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    intrinsic = intrinsic_call(S, K, T, r)
    time_value = y_true - intrinsic
    mask_tv = time_value > 0.05        # meaningful time value
    n_tv = int(mask_tv.sum())
    print(f"\n[B] Intrinsic-filtered  (time value > 0.05,  n={n_tv}/{n})")
    if n_tv > 0:
        abs_err_tv = np.abs(y_nn[mask_tv] - y_true[mask_tv])
        rel_err_tv = abs_err_tv / y_true[mask_tv]
        print(f"    Mean abs err      : {abs_err_tv.mean():.5f}")
        print(f"    Mean rel err      : {rel_err_tv.mean()*100:7.2f}%")
        print(f"    P95 rel err       : {np.quantile(rel_err_tv, 0.95)*100:7.2f}%")

    # =========================================================================
    # C. IV-space error (industry standard)
    # =========================================================================
    print(f"\n[C] IV-space error  (via Brent solver)")
    # Solve IV only on intrinsic-filtered subset for speed + numerical robustness
    X_sub = X[mask_tv]
    y_true_sub = y_true[mask_tv]
    y_nn_sub = y_nn[mask_tv]

    iv_true = iv_batch(y_true_sub, X_sub)
    iv_nn = iv_batch(y_nn_sub, X_sub)
    valid = ~(np.isnan(iv_true) | np.isnan(iv_nn))
    n_iv = int(valid.sum())

    iv_diff = np.abs(iv_nn[valid] - iv_true[valid])
    print(f"    Solvable samples  : {n_iv}/{n_tv}  ({n_iv/max(n_tv,1)*100:.1f}%)")
    if n_iv > 0:
        print(f"    Mean |Δσ|         : {_fmt_vol_pts(iv_diff.mean())}")
        print(f"    P95 |Δσ|          : {_fmt_vol_pts(np.quantile(iv_diff, 0.95))}")
        print(f"    Max |Δσ|          : {_fmt_vol_pts(iv_diff.max())}")

    # =========================================================================
    # D. Moneyness-stratified report
    # =========================================================================
    print(f"\n[D] Moneyness-stratified (IV error)")
    print(f"    {'bucket':<10s}{'n':>6s}{'mean |Δσ|':>14s}{'p95 |Δσ|':>14s}"
          f"{'mean rel (price)':>20s}")
    for name, mask in moneyness_buckets(X):
        mask_sub = mask & mask_tv
        cnt = int(mask_sub.sum())
        if cnt == 0:
            print(f"    {name:<10s}{cnt:>6d}{'-':>14s}{'-':>14s}{'-':>20s}")
            continue
        # IV error for this bucket
        Xb = X[mask_sub]
        iv_t = iv_batch(y_true[mask_sub], Xb)
        iv_p = iv_batch(y_nn[mask_sub], Xb)
        v = ~(np.isnan(iv_t) | np.isnan(iv_p))
        diff = np.abs(iv_p[v] - iv_t[v]) if v.any() else np.array([])

        # Price-space rel err
        rel = np.abs(y_nn[mask_sub] - y_true[mask_sub]) / y_true[mask_sub]

        diff_mean = f"{_fmt_vol_pts(diff.mean())}" if len(diff) else "n/a"
        diff_p95 = f"{_fmt_vol_pts(np.quantile(diff, 0.95))}" if len(diff) else "n/a"
        print(f"    {name:<10s}{cnt:>6d}{diff_mean:>14s}{diff_p95:>14s}"
              f"{_fmt_pct(rel.mean()):>20s}")

    # =========================================================================
    # E. Speed comparison
    # =========================================================================
    print(f"\n[E] Inference speed  (CPU, n={n})")
    print(f"    BSM closed-form   : {t_bsm*1000:7.1f} ms  ({t_bsm/n*1e6:6.1f} µs/opt)")
    print(f"    NN Pricer         : {t_nn*1000:7.1f} ms  ({t_nn/n*1e6:6.1f} µs/opt)")
    print(f"    NN speedup        : {t_bsm/t_nn:.1f}×")

    # =========================================================================
    # F. ATM point check
    # =========================================================================
    atm = np.array([[100.0, 100.0, 1.0, 0.03, 0.0, 0.2]])
    bsm_atm = call_price(BSMInputs(*atm[0]))
    nn_atm = price_batch(model, atm, device=dev)[0]
    err_pct = abs(bsm_atm - nn_atm) / bsm_atm * 100
    print(f"\n[F] ATM point check  (S=K=100, T=1, r=3%, σ=20%)")
    print(f"    BSM               : {bsm_atm:.4f}")
    print(f"    NN                : {nn_atm:.4f}")
    print(f"    diff              : {abs(bsm_atm - nn_atm):.4f}  ({err_pct:.2f}%)")

    return {
        "iv_mean": float(iv_diff.mean()) if n_iv > 0 else float("nan"),
        "iv_p95": float(np.quantile(iv_diff, 0.95)) if n_iv > 0 else float("nan"),
        "atm_err_pct": err_pct,
        "speedup": t_bsm / t_nn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/nn_pricer.pt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--label", default="", help="Tag printed in header (e.g., 'BEFORE', 'AFTER')")
    args = ap.parse_args()
    run(args.model, args.n, args.seed, args.label)


if __name__ == "__main__":
    main()
