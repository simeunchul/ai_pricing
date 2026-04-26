"""B2 → ELS Daily NAV demo (#1 통합).

한화 8286호 (3-asset KOSPI200/S&P500/SX5E) 의 fair value 를 매일 자동 산출.
3개 자산의 IV surface 를 받아 B2 로 각각 calibrate → vol 추출 → Layer A 재가격.

KOSPI200/SX5E 의 옵션 IV 직접 받기 어려워서 SPY 의 IV surface 를 모든 자산에
동일하게 적용 (proxy). production 에선 자산별 surface 필요.

Usage:
  python scripts/fetch_spy_iv.py    # 최신 SPY IV 받기 (사전)
  python scripts/integrate_b2_els_daily.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from pricing.els.step_down import StepDownELS
from ai_pricing.integrations.els_daily_nav import els_daily_nav


def load_spy_iv_surface(snap_path: str) -> np.ndarray:
    """Load SPY IV snapshot → align to model's 25-cell grid."""
    from ai_pricing.deep_calib.surface import STRIKES, MATURITIES
    snap = json.loads(Path(snap_path).read_text(encoding="utf-8"))
    market = np.array(snap["iv_grid"])
    snap_T = np.array(snap["maturities_months"]) / 12
    snap_m = np.array(snap["moneyness"])

    aligned = np.full((len(MATURITIES), len(STRIKES)), np.nan)
    for i, T in enumerate(MATURITIES):
        i_src = int(np.argmin(np.abs(snap_T - T)))
        for j, m in enumerate(STRIKES):
            j_src = int(np.argmin(np.abs(snap_m - m)))
            aligned[i, j] = market[i_src, j_src]
    return aligned.flatten(), snap


def main():
    print("=" * 72)
    print("  B2 → ELS Daily NAV — 한화 8286호 자동 재가격")
    print("=" * 72)

    # 1. Latest SPY IV surface
    snaps = sorted((ROOT / "data" / "market_snapshots").glob("spy_iv_*.json"))
    snaps = [s for s in snaps if "_calib" not in s.name]
    if not snaps:
        print("No SPY snapshot found. Run: python scripts/fetch_spy_iv.py")
        return
    snap_path = str(snaps[-1])
    iv_25, snap_info = load_spy_iv_surface(snap_path)
    print(f"\n[market] SPY snapshot: {snap_info['snapshot_time']}")
    print(f"         spot={snap_info['spot']:.2f}, valid cells: 25/25")

    # 2. 한화 8286호 product spec
    product = StepDownELS(
        S0=np.array([100.0, 100.0, 100.0]),
        barriers=[0.95, 0.90, 0.85, 0.85, 0.80, 0.75],
        ki_barrier=0.50,
        coupon_rate=0.1131 / 2,    # 연 11.31% / 반기 = 5.655% per period
        maturity_years=3.0,
        obs_per_year=2,
        notional=10_000.0,
    )
    asset_names = ["KOSPI200", "S&P500", "SX5E"]
    print(f"\n[product] 한화스마트ELS 제8286호 (3-asset worst-of, 3Y, 연 11.31%)")
    print(f"          assets: {asset_names}")
    print(f"          notional: {product.notional:,} KRW")

    # 3. Per-asset IV surface (SPY proxy for all 3 assets)
    iv_per_asset = [iv_25, iv_25, iv_25]  # production: 자산별 다른 IV

    # 4. Daily NAV via B2 → Layer A
    print(f"\n[calibration] B2 Deep Calib 실행 (3 assets × NN gradient descent)...")
    res = els_daily_nav(
        product=product,
        iv_surfaces_per_asset=iv_per_asset,
        asset_names=asset_names,
        r=0.035,
        q=np.array([0.017, 0.015, 0.031]),
        corr=np.array([
            [1.00, 0.45, 0.55],
            [0.45, 1.00, 0.70],
            [0.55, 0.70, 1.00],
        ]),
        market_date=snap_info["snapshot_time"][:10],
        n_paths=50_000,
        n_steps_per_year=252,
        seed=2026,
    )

    # 5. Report
    print(f"\n[result]")
    print(f"  market date              : {res.market_date}")
    print(f"  per-asset σ (calibrated) : {[f'{s*100:.2f}%' for s in res.sigmas_used]}")
    print(f"  per-asset IV RMSE (vp)   : {[f'{r*100:.2f}' for r in res.iv_rmse_per_asset]}")
    print()
    print(f"  발행가 (notional)        : {res.issue_price_krw:>10,.1f} KRW")
    print(f"  오늘 fair value (NAV)    : {res.fair_value_krw:>10,.1f} ± {res.fair_value_stderr:.1f} KRW")
    print(f"  vs 발행가                : {res.deviation_pct:+.2f}%")
    print(f"  KI hit prob              : {res.ki_hit_prob*100:.1f}%")
    print(f"  expected life            : {res.expected_life_years:.2f} years")

    # 6. Save
    out = ROOT / "data" / "els_samples" / "daily_nav_8286.json"
    out.write_text(json.dumps({
        "market_date": res.market_date,
        "snapshot_time": snap_info["snapshot_time"],
        "fair_value_krw": res.fair_value_krw,
        "fair_value_stderr": res.fair_value_stderr,
        "issue_price_krw": res.issue_price_krw,
        "deviation_pct": res.deviation_pct,
        "sigmas_used": res.sigmas_used,
        "asset_names": res.asset_names,
        "iv_rmse_per_asset": res.iv_rmse_per_asset,
        "ki_hit_prob": res.ki_hit_prob,
        "expected_life_years": res.expected_life_years,
        "interpretation": (
            f"발행가 10,000 대비 {res.deviation_pct:+.2f}%. "
            f"calibrated σ {[round(s*100,1) for s in res.sigmas_used]}% 환경 기준."
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
