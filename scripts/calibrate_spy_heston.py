"""실시장 SPY IV surface → Heston 5-param Deep Calibration.

플랜의 "B2 실데이터 calibration" 단계.

Pipeline:
  1. data/market_snapshots/spy_iv_*.json 로드
  2. 학습된 DeepCalibNet (3k Heston synthetic 학습본) 사용
  3. gradient descent on param space → fitted HestonParams
  4. fitted params → semi-analytic IV surface 재생성
  5. 시장 IV vs fit IV RMSE 측정
  6. semi-analytic Nelder-Mead 와 속도/정확도 비교

Usage:
  python scripts/calibrate_spy_heston.py
  python scripts/calibrate_spy_heston.py --snapshot data/market_snapshots/spy_iv_20260425.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from pricing.heston import HestonParams
from ai_pricing.deep_calib.surface import (
    iv_surface, STRIKES, MATURITIES, N_POINTS,
)
from ai_pricing.deep_calib.model import DeepCalibNet, DeepCalibConfig
from ai_pricing.deep_calib.calibrate import calibrate as nn_calibrate


def load_market_iv(path: str) -> tuple[np.ndarray, dict]:
    """Load JSON snapshot and align to model's STRIKES × MATURITIES grid."""
    snap = json.loads(Path(path).read_text(encoding="utf-8"))
    market_grid = np.array(snap["iv_grid"])             # (5 maturities × 5 moneyness)
    snap_T = np.array(snap["maturities_months"]) / 12   # to years
    snap_m = np.array(snap["moneyness"])                # K/S

    # Model's STRIKES are also K/S. Realign by nearest neighbor.
    aligned = np.full((len(MATURITIES), len(STRIKES)), np.nan)
    for i, T in enumerate(MATURITIES):
        i_src = int(np.argmin(np.abs(snap_T - T)))
        for j, m in enumerate(STRIKES):
            j_src = int(np.argmin(np.abs(snap_m - m)))
            aligned[i, j] = market_grid[i_src, j_src]

    return aligned.flatten(), snap


def heston_iv_surface_via_semi(p: HestonParams) -> np.ndarray:
    """Build full 25-IV surface via semi-analytic Heston (slow ground truth)."""
    return iv_surface(p, S=1.0, r=0.045)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None,
                    help="JSON path; auto-pick latest if omitted")
    ap.add_argument("--model", default="models/deep_calib.pt")
    args = ap.parse_args()

    snap_path = args.snapshot
    if snap_path is None:
        snaps = sorted(Path("data/market_snapshots").glob("*_iv_*.json"))
        if not snaps:
            print("No snapshot found. Run: python scripts/fetch_spy_iv.py")
            return
        snap_path = str(snaps[-1])
    print(f"[calib] snapshot: {snap_path}")

    # 1. Load market IV (align to model grid)
    iv_market, snap = load_market_iv(snap_path)
    print(f"[calib] {snap['ticker']} spot={snap['spot']:.2f} r={snap['r']}")
    print(f"[calib] market IV stats: min={iv_market.min():.3f} "
          f"mean={iv_market.mean():.3f} max={iv_market.max():.3f}")

    # 2. Load DeepCalibNet
    import torch
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = DeepCalibNet(DeepCalibConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3. Deep Calib (NN gradient descent)
    t0 = time.perf_counter()
    p_dc, rmse_dc = nn_calibrate(
        iv_market, model,
        ckpt["x_mean"], ckpt["x_std"],
        ckpt["y_mean"], ckpt["y_std"],
        lr=5e-2, steps=400,
    )
    t_dc = time.perf_counter() - t0

    # 4. Verify with true Heston semi-analytic
    iv_fit_semi = heston_iv_surface_via_semi(p_dc)
    valid = ~np.isnan(iv_fit_semi)
    rmse_semi = float(np.sqrt(np.nanmean((iv_fit_semi[valid] - iv_market[valid]) ** 2)))

    # 5. Also try classical Nelder-Mead for speed comparison
    from scipy.optimize import minimize
    def loss(x):
        try:
            p = HestonParams(*x)
            grid = heston_iv_surface_via_semi(p)
            return float(np.nanmean((grid - iv_market) ** 2))
        except Exception:
            return 1e6

    t0 = time.perf_counter()
    res_cl = minimize(loss, x0=[2.0, 0.05, 0.4, -0.5, 0.05],
                      method="Nelder-Mead", options={"maxiter": 200})
    t_cl = time.perf_counter() - t0
    p_cl = HestonParams(*res_cl.x)
    iv_cl = heston_iv_surface_via_semi(p_cl)
    rmse_cl = float(np.sqrt(np.nanmean((iv_cl - iv_market) ** 2)))

    # 6. Print
    print()
    print("=" * 70)
    print(f"Method               κ      θ      ξ      ρ       v₀     RMSE(semi)  Wall")
    print("-" * 70)
    print(f"Deep Calib (NN)   {p_dc.kappa:6.3f} {p_dc.theta:6.4f} {p_dc.xi:6.3f} "
          f"{p_dc.rho:+6.3f} {p_dc.v0:6.4f}    {rmse_semi*100:>5.2f} vp   {t_dc*1000:>6.1f}ms")
    print(f"Nelder-Mead       {p_cl.kappa:6.3f} {p_cl.theta:6.4f} {p_cl.xi:6.3f} "
          f"{p_cl.rho:+6.3f} {p_cl.v0:6.4f}    {rmse_cl*100:>5.2f} vp   {t_cl*1000:>6.1f}ms")
    print("=" * 70)
    print(f"Speedup: {t_cl / t_dc:.1f}× (Deep Calib vs Nelder-Mead)")
    print(f"Accuracy gap: {(rmse_semi - rmse_cl)*100:+.2f} vp")

    out = Path(snap_path).with_name(Path(snap_path).stem + "_calib.json")
    out.write_text(json.dumps({
        "snapshot": snap_path,
        "ticker": snap["ticker"],
        "deep_calib": {
            "params": {"kappa": p_dc.kappa, "theta": p_dc.theta, "xi": p_dc.xi,
                       "rho": p_dc.rho, "v0": p_dc.v0},
            "rmse_vp": rmse_semi * 100,
            "wall_ms": t_dc * 1000,
        },
        "nelder_mead": {
            "params": {"kappa": p_cl.kappa, "theta": p_cl.theta, "xi": p_cl.xi,
                       "rho": p_cl.rho, "v0": p_cl.v0},
            "rmse_vp": rmse_cl * 100,
            "wall_ms": t_cl * 1000,
        },
        "speedup": float(t_cl / t_dc),
    }, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
