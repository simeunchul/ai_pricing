"""Layer A · 실제 한화 ELS 공시 기반 재가격.

Target: 한화스마트ELS 제8286호 (DART rcept_no=20260424000266)
  - 기초자산: KOSPI200, S&P500, SX5E (3자산 worst-of)
  - 만기: 3년 (2026-05-06 → 2029-05-09)
  - 쿠폰: 연 11.31% (세전)
  - 발행가: 10,000원

미확정 스펙 (DART XML 표 구조 분해 실패 → 한국 step-down ELS 업계 표준 가정):
  - KI barrier: 50%
  - Autocall barriers: 95% - 90% - 85% - 85% - 80% - 75%  (반기 관찰 6회)
  - 관찰주기: 6개월

Usage:
    python scripts/reprice_hanwha_els.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pricing.els.step_down import StepDownELS, price_els


# --- 공시 스펙 -----------------------------------------------------------------
ISSUE_NO = 8286
ISSUE_DATE = "2026-05-06"
MATURITY_DATE = "2029-05-09"
NOTIONAL_PER_UNIT = 10_000
COUPON_PER_YEAR = 0.1131           # 연 11.31%
MATURITY_YEARS = 3.0

# --- 업계 표준 구조 가정 --------------------------------------------------------
KI_BARRIER_PCT = 50                # 만기 시 원금손실 시작선
AUTOCALL_SCHEDULE = [0.95, 0.90, 0.85, 0.85, 0.80, 0.75]   # 6M×6 = 3Y
OBS_PER_YEAR = 2

# --- 시장 파라미터 (대표치, 실제 calibration 은 B2 Deep Calib 영역) --------------
MARKET = {
    "r": 0.035,                    # 3년물 국고채
    "underlyings": ["KOSPI200", "S&P500", "SX5E"],
    "sigma": np.array([0.22, 0.18, 0.22]),   # 3자산 implied vol (대표)
    "q":     np.array([0.017, 0.015, 0.031]),  # dividend yield
    "corr":  np.array([
        [1.00, 0.45, 0.55],
        [0.45, 1.00, 0.70],
        [0.55, 0.70, 1.00],
    ]),
}

# --- 설정 ----------------------------------------------------------------------
N_PATHS = 50_000
N_STEPS_PER_YEAR = 252


def semi_annual_coupon(annual: float, obs_per_year: int) -> float:
    """Convert annual to per-period coupon (simple)."""
    return annual / obs_per_year


def main():
    product = StepDownELS(
        S0=np.array([100.0, 100.0, 100.0]),        # 3자산, 상대값
        barriers=AUTOCALL_SCHEDULE,
        ki_barrier=KI_BARRIER_PCT / 100,
        coupon_rate=semi_annual_coupon(COUPON_PER_YEAR, OBS_PER_YEAR),
        maturity_years=MATURITY_YEARS,
        obs_per_year=OBS_PER_YEAR,
        notional=NOTIONAL_PER_UNIT,
    )

    print(f"=== 한화스마트ELS 제{ISSUE_NO}호 재가격 ===")
    print(f"  발행 {ISSUE_DATE} · 만기 {MATURITY_DATE} · {MATURITY_YEARS:.0f}Y")
    print(f"  기초자산 {MARKET['underlyings']}")
    print(f"  연 쿠폰 {COUPON_PER_YEAR*100:.2f}%   반기당 {semi_annual_coupon(COUPON_PER_YEAR, OBS_PER_YEAR)*100:.3f}%")
    print(f"  KI {KI_BARRIER_PCT}%  autocall {[int(b*100) for b in AUTOCALL_SCHEDULE]}")
    print(f"  공시 발행가 {NOTIONAL_PER_UNIT:,}원\n")

    # Run with and without corr sensitivities
    scenarios = {
        "base (대표 vol)": (MARKET["sigma"], MARKET["corr"]),
        "vol +3% shock":   (MARKET["sigma"] + 0.03, MARKET["corr"]),
        "vol -3% shock":   (MARKET["sigma"] - 0.03, MARKET["corr"]),
        "corr → 0.9":      (MARKET["sigma"], np.full((3, 3), 0.9) + (1 - 0.9) * np.eye(3)),
        "corr → 0.2":      (MARKET["sigma"], np.full((3, 3), 0.2) + (1 - 0.2) * np.eye(3)),
    }

    rows = []
    t0 = time.perf_counter()
    for name, (sig, corr) in scenarios.items():
        res = price_els(
            product,
            r=MARKET["r"],
            q=MARKET["q"],
            sigma=sig,
            corr=corr,
            n_paths=N_PATHS,
            n_steps_per_year=N_STEPS_PER_YEAR,
            seed=2026,
        )
        dev_pct = (res.price - NOTIONAL_PER_UNIT) / NOTIONAL_PER_UNIT * 100
        rows.append({
            "scenario": name,
            "price_krw": round(res.price, 1),
            "stderr_krw": round(res.stderr, 1),
            "ki_hit_prob": round(res.ki_hit_prob, 4),
            "expected_life_years": round(res.expected_life, 3),
            "vs_notional_pct": round(dev_pct, 2),
            "autocall_prob_by_period": [round(p, 3) for p in res.autocall_prob],
        })
        print(f"  [{name:<16s}] price {res.price:>9,.1f} ±{res.stderr:>5.1f}  "
              f"KI {res.ki_hit_prob*100:>5.1f}%  E[life] {res.expected_life:.2f}y  "
              f"vs notional {dev_pct:+6.2f}%")
    print(f"\n  (total {time.perf_counter()-t0:.1f}s, {N_PATHS:,} paths × {N_STEPS_PER_YEAR}/y)")

    out = Path("data/els_samples/reprice_hanwha_8286.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "issue_no": ISSUE_NO,
        "notional_krw": NOTIONAL_PER_UNIT,
        "coupon_pct": COUPON_PER_YEAR * 100,
        "scenarios": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
