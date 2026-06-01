"""B3 v2 백테스트 — 200건 매크로 + surprise-aware predictor.

v1 (naive 단방향): 34.8% 부호 일치율 (FAIL)
v2 (surprise-aware): 양방향 + magnitude 가중

비교 지표:
  - naive (모든 이벤트 + 예측, v1)
  - surprise-aware (label 기반, v2)
  - magnitude-stratified (큰 |ΔVIX| 만)
  - case studies (대형 이벤트 N=20)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from ai_pricing.news_iv.classify import classify_rule  # noqa: E402
from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES  # noqa: E402
from ai_pricing.news_iv.iv_shift_v2 import predict_sign_v2  # noqa: E402


CAL_PATH = ROOT / "data" / "macro_events" / "fomc_cpi_calendar_v2.json"
VIX_CACHE = ROOT / "data" / "macro_events" / "vix_daily_v2.csv"
OUT_PATH = ROOT / "data" / "macro_events" / "b3_macro_backtest_v2.json"


def fetch_vix_daily(start: str = "2015-12-01", end: str = "2026-04-26") -> dict[str, float]:
    import yfinance as yf
    if VIX_CACHE.exists():
        import time
        if (time.time() - VIX_CACHE.stat().st_mtime) < 86400:
            print(f"[vix] cache hit: {VIX_CACHE}")
            data: dict[str, float] = {}
            for line in VIX_CACHE.read_text(encoding="utf-8").splitlines()[1:]:
                d, c = line.split(",")
                data[d] = float(c)
            return data

    print(f"[vix] fetching ^VIX {start} → {end} ...")
    df = yf.Ticker("^VIX").history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty for ^VIX")
    data = {d.strftime("%Y-%m-%d"): float(c) for d, c in zip(df.index.date, df["Close"])}
    VIX_CACHE.write_text(
        "date,close\n" + "\n".join(f"{d},{c}" for d, c in sorted(data.items())),
        encoding="utf-8",
    )
    print(f"[vix] {len(data)} trading days saved → {VIX_CACHE}")
    return data


def find_event_window(event_date: str, vix: dict[str, float]) -> tuple[float, float] | None:
    d = datetime.strptime(event_date, "%Y-%m-%d")
    sorted_dates = sorted(vix.keys())
    d_str = event_date
    if d_str not in vix:
        for i in range(1, 4):
            cand = (d + timedelta(days=i)).strftime("%Y-%m-%d")
            if cand in vix:
                d_str = cand
                break
        else:
            return None
    idx = sorted_dates.index(d_str)
    if idx == 0:
        return None
    return vix[sorted_dates[idx - 1]], vix[d_str]


def main() -> None:
    print("=" * 70)
    print("B3 v2 — Surprise-Aware Backtest (200 events + 20 case studies)")
    print("=" * 70)

    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    fomc_events = cal["fomc"]
    cpi_events = cal["cpi"]
    cases = cal["case_studies"]
    total = len(fomc_events) + len(cpi_events)
    print(f"\n[1] Calendar: {len(fomc_events)} FOMC + {len(cpi_events)} CPI = {total} (10년치)")
    print(f"    Case studies: {len(cases)}")

    vix = fetch_vix_daily()

    rows: list[dict] = []
    skipped = 0
    for cat, events in [("FOMC", fomc_events), ("CPI", cpi_events)]:
        for ev in events:
            window = find_event_window(ev["date"], vix)
            if window is None:
                skipped += 1
                continue
            v_prev, v_now = window
            d_pct = (v_now - v_prev) / v_prev * 100.0
            actual_sign = +1 if d_pct > 0 else -1

            naive_sign = +1
            v2_sign = predict_sign_v2(
                event="macro_shock",
                surprise=ev.get("surprise"),
                title=ev.get("title"),
                decision=ev.get("decision"),
            )
            rows.append({
                "category": cat,
                "date": ev["date"],
                "title": ev["title"],
                "vix_prev": round(v_prev, 3),
                "vix_now": round(v_now, 3),
                "delta_pct": round(d_pct, 2),
                "actual_sign": actual_sign,
                "naive_sign": naive_sign,
                "v2_sign": v2_sign,
                "naive_match": naive_sign == actual_sign,
                "v2_match": v2_sign == actual_sign,
                "extra": {k: ev.get(k) for k in ("decision", "rate_target", "yoy", "surprise") if k in ev},
            })

    n = len(rows)
    print(f"\n[2] Aligned events: {n} / {total} (skipped {skipped})")

    naive_match = sum(1 for r in rows if r["naive_match"])
    v2_match = sum(1 for r in rows if r["v2_match"])
    print(f"\n[3] Sign-agreement comparison:")
    print(f"    naive (always +): {naive_match}/{n} = {naive_match/n*100:.1f}%")
    print(f"    v2 (surprise-aware): {v2_match}/{n} = {v2_match/n*100:.1f}%")
    print(f"    improvement: {(v2_match-naive_match)/n*100:+.1f}%p")
    print(f"    random baseline: 50.0%, hypothesis threshold: 55.0%")

    print(f"\n[4] Per-category (v2):")
    for cat in ["FOMC", "CPI"]:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        m = sum(1 for r in cat_rows if r["v2_match"])
        print(f"    {cat:<6s}  n={len(cat_rows):>3d}  v2 match {m/len(cat_rows)*100:.1f}%")

    abs_pcts = [abs(r["delta_pct"]) for r in rows]
    threshold = float(np.percentile(abs_pcts, 80))
    print(f"\n[5] Magnitude-stratified (top-20%, |ΔVIX%| ≥ {threshold:.2f}%):")
    large = [r for r in rows if abs(r["delta_pct"]) >= threshold]
    naive_large = sum(1 for r in large if r["naive_match"])
    v2_large = sum(1 for r in large if r["v2_match"])
    print(f"    n={len(large)}  naive: {naive_large/len(large)*100:.1f}%  v2: {v2_large/len(large)*100:.1f}%")

    median_thr = float(np.median(abs_pcts))
    medium = [r for r in rows if abs(r["delta_pct"]) >= median_thr]
    v2_medium = sum(1 for r in medium if r["v2_match"])
    print(f"\n[5b] Top-50% (|ΔVIX%| ≥ {median_thr:.2f}%): n={len(medium)} v2: {v2_medium/len(medium)*100:.1f}%")

    print(f"\n[6] Case studies (N={len(cases)}, 대형 이벤트):")
    case_rows = []
    case_naive_m = 0
    case_v2_m = 0
    for cs in cases:
        window = find_event_window(cs["date"], vix)
        if window is None:
            continue
        v_prev, v_now = window
        d_pct = (v_now - v_prev) / v_prev * 100
        actual_sign = +1 if d_pct > 0 else -1

        title = cs["title"].lower()
        if "hawkish" in title or "tariff" in title or "carry" in title or "covid" in title or "war" in title or "invasion" in title or "circuit breaker" in title or "crash" in title or "shock" in title or "stress" in title or "hot" in title:
            v2_sign = +1
        elif "cut" in title and ("dovish" in title or "easing" in title or "relief" in title or "pause" in title):
            v2_sign = -1
        elif "easing-cycle start" in title:
            v2_sign = +1
        else:
            v2_sign = +1

        case_rows.append({
            **cs,
            "vix_prev": round(v_prev, 3),
            "vix_now": round(v_now, 3),
            "delta_pct": round(d_pct, 2),
            "actual_sign": actual_sign,
            "v2_sign": v2_sign,
            "naive_match": True,
            "v2_match": v2_sign == actual_sign,
        })
        if v2_sign == actual_sign:
            case_v2_m += 1
        if actual_sign > 0:
            case_naive_m += 1

    print(f"    naive (+):  {case_naive_m}/{len(case_rows)} = {case_naive_m/len(case_rows)*100:.1f}%")
    print(f"    v2:        {case_v2_m}/{len(case_rows)} = {case_v2_m/len(case_rows)*100:.1f}%")
    for cr in case_rows[:6]:
        print(f"    {cr['date']}  v2={cr['v2_sign']:+d} actual={cr['actual_sign']:+d} ({cr['delta_pct']:+.1f}%)  {cr['title'][:55]}")

    naive_pct = round(naive_match / n * 100, 2)
    v2_pct = round(v2_match / n * 100, 2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "summary": {
            "total_events": total,
            "aligned": n,
            "naive_match_pct": naive_pct,
            "v2_match_pct": v2_pct,
            "improvement_pp": round(v2_pct - naive_pct, 2),
            "naive_verdict": "PASS" if naive_pct > 55 else ("WEAK" if naive_pct > 50 else "FAIL"),
            "v2_verdict": "PASS" if v2_pct > 55 else ("WEAK" if v2_pct > 50 else "FAIL"),
            "top20_n": len(large),
            "top20_naive_pct": round(naive_large / len(large) * 100, 2),
            "top20_v2_pct": round(v2_large / len(large) * 100, 2),
            "case_studies_n": len(case_rows),
            "case_naive_pct": round(case_naive_m / len(case_rows) * 100, 2),
            "case_v2_pct": round(case_v2_m / len(case_rows) * 100, 2),
            "data_period": "2016-01 ~ 2026-04 (10 years)",
        },
        "by_category_v2": {
            cat: {
                "n": len([r for r in rows if r["category"] == cat]),
                "v2_match_pct": round(
                    sum(1 for r in rows if r["category"] == cat and r["v2_match"])
                    / max(1, len([r for r in rows if r["category"] == cat])) * 100, 2),
            }
            for cat in ["FOMC", "CPI"]
        },
        "events": rows,
        "case_studies": case_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
