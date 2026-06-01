"""한화투자증권 ELS/ELB 추가 표본 재가격 (1057호 외).

이전 표본 (8286 호) 은 3자산 worst-of step-down. 새 표본 3건은 모두
삼성전자 단일자산 단기물 (1~2년) → step-down 가정 못 씀.

단일자산 1Y/2Y ELB 의 통상 구조:
  - 만기 시 KI 50% 미터치 → 원금 + 누적쿠폰
  - KI 터치 후 만기 종가 < 행사 → 원금 손실
  - 사실상 "knock-in put" 매도 + 쿠폰 받기

여기서는 단일자산 step-down (autocall 95-90-85-...) 으로 간략화하여
KI=50%, autocall 6M 마다 가정 후 재가격.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))

from pricing.els.step_down import StepDownELS, price_els


def reprice_single_asset(
    issue_no: int,
    maturity_years: float,
    coupon_pct_per_year: float,
    sigma: float = 0.30,         # 삼성전자 대표 vol (개별주)
    q: float = 0.025,
    r: float = 0.035,
    obs_per_year: int = 2,
    n_paths: int = 30_000,
) -> dict:
    n_obs = int(obs_per_year * maturity_years)
    autocall = [round(0.95 - 0.05 * i, 2) for i in range(n_obs - 1)] + [0.75]

    product = StepDownELS(
        S0=np.array([100.0]),
        barriers=autocall,
        ki_barrier=0.50,
        coupon_rate=coupon_pct_per_year / 100 / obs_per_year,
        maturity_years=maturity_years,
        obs_per_year=obs_per_year,
        notional=10_000,
    )

    res = price_els(
        product,
        r=r,
        q=np.array([q]),
        sigma=np.array([sigma]),
        corr=np.array([[1.0]]),
        n_paths=n_paths,
        n_steps_per_year=252,
        seed=2026 + issue_no,
    )
    dev_pct = (res.price - 10_000) / 10_000 * 100
    return {
        "issue_no": issue_no,
        "maturity_years": maturity_years,
        "coupon_pct_per_year": coupon_pct_per_year,
        "sigma_used": sigma,
        "autocall_schedule": autocall,
        "ki_barrier_pct": 50,
        "price_krw": round(res.price, 1),
        "stderr_krw": round(res.stderr, 1),
        "ki_hit_prob": round(res.ki_hit_prob, 4),
        "expected_life_years": round(res.expected_life, 3),
        "vs_notional_pct": round(dev_pct, 2),
    }


def main():
    print("=" * 70)
    print("  한화투자증권 ELS/ELB 추가 표본 재가격")
    print("=" * 70)

    samples_meta = json.loads(
        (ROOT / "data" / "els_samples" / "parsed_dart.json").read_text(encoding="utf-8"))

    rows = []
    for s in samples_meta:
        if s["issue_no"] is None or s["coupon_rate_pct_per_year"] is None:
            print(f"\n[#{s.get('issue_no')}] 쿠폰 미추출 — 메타데이터까지만 확보 (재가격 skip)")
            rows.append({
                "issue_no": s.get("issue_no"),
                "underlyings": s.get("underlyings"),
                "issue_date": s.get("issue_date"),
                "maturity_date": s.get("maturity_date"),
                "issue_price_krw": s.get("issue_price_krw"),
                "coupon_rate_pct_per_year": None,
                "status": "meta-only (쿠폰 미추출)",
                "_file": s.get("_file"),
            })
            continue

        issue_date = s.get("issue_date")
        maturity_date = s.get("maturity_date")
        if issue_date and maturity_date:
            from datetime import datetime
            i = datetime.strptime(issue_date, "%Y-%m-%d")
            m = datetime.strptime(maturity_date, "%Y-%m-%d")
            years = (m - i).days / 365.0
        else:
            years = 2.0

        print(f"\n[#{s['issue_no']}] {s.get('underlyings')} · {years:.2f}Y · {s['coupon_rate_pct_per_year']}%/yr")
        t0 = time.perf_counter()
        result = reprice_single_asset(
            issue_no=s["issue_no"],
            maturity_years=years,
            coupon_pct_per_year=s["coupon_rate_pct_per_year"],
        )
        result["underlyings"] = s.get("underlyings")
        result["issue_date"] = issue_date
        result["maturity_date"] = maturity_date
        result["status"] = "priced"
        result["_file"] = s.get("_file")
        rows.append(result)
        print(f"    price {result['price_krw']:>9,.1f}원  KI {result['ki_hit_prob']*100:.1f}%  "
              f"E[life] {result['expected_life_years']:.2f}y  "
              f"vs notional {result['vs_notional_pct']:+.2f}%  "
              f"({time.perf_counter()-t0:.1f}s)")

    out = ROOT / "data" / "els_samples" / "additional_repriced.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {out}")

    priced = [r for r in rows if r.get("status") == "priced"]
    if priced:
        devs = [abs(r["vs_notional_pct"]) for r in priced]
        print(f"\n[Summary] {len(priced)}건 재가격, |편차| 평균 {np.mean(devs):.2f}%, max {np.max(devs):.2f}%")
        within_3pct = sum(1 for d in devs if d <= 3.0)
        print(f"          ±3% 이내: {within_3pct}/{len(priced)}건")

    print(f"\n[Combined with 8286호]:")
    rep_8286 = json.loads((ROOT / "data" / "els_samples" / "reprice_hanwha_8286.json").read_text(encoding="utf-8"))
    base_8286 = next(r for r in rep_8286["scenarios"] if r["scenario"] == "base (대표 vol)")
    print(f"    #8286 (3자산 worst-of, 3Y, 11.31%): vs notional {base_8286['vs_notional_pct']:+.2f}%")
    for r in priced:
        print(f"    #{r['issue_no']} ({'/'.join(r['underlyings'])}, {r['maturity_years']:.1f}Y, {r['coupon_pct_per_year']:.2f}%): vs notional {r['vs_notional_pct']:+.2f}%")

    all_devs = [abs(base_8286["vs_notional_pct"])] + [abs(r["vs_notional_pct"]) for r in priced]
    all_within = sum(1 for d in all_devs if d <= 3.0)
    print(f"\n  Total samples: {len(all_devs)}건, |편차| 평균 {np.mean(all_devs):.2f}%, ±3% 이내 {all_within}/{len(all_devs)}건")


if __name__ == "__main__":
    main()
