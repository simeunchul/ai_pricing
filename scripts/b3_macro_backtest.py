"""B3 News-IV 매크로 이벤트 백테스트 (옵션 C 1단계).

가설: rule classifier 가 macro_shock 으로 분류한 이벤트 (FOMC + CPI 발표) 직후
      VIX 의 1일 변화 부호가 IV_SHIFT_RULES["macro_shock"] = +0.05 (즉 양수) 와
      유의하게 일치하는가?

데이터:
  - 매크로 이벤트 캘린더: data/macro_events/fomc_cpi_calendar.json
    (FOMC 18건 2024-01 ~ 2026-03, BLS CPI 28건 2024-01 ~ 2026-04)
  - VIX 일별 close: yfinance ^VIX (실시간 fetch + 캐시)

방법:
  1. 이벤트일 d 가 영업일 (NYSE) 인지 확인
  2. ΔVIX = VIX(d) - VIX(d-1)   ← FOMC 는 보통 미국 동부 14:00 발표, 종가 close 에 반영
     Δ% = ΔVIX / VIX(d-1) * 100
  3. 가설:
     - macro_shock prediction = +0.05 (양수)
     - 부호 일치 = (sign(ΔVIX) > 0)
  4. 별도 카테고리:
     - FOMC vs CPI 분리
     - "hawkish surprise" (sticky CPI hot, FOMC hawkish dot plot) 만 더 강한 +shift 예상
  5. 검증선: 50% 랜덤 baseline, 55% 검증, 60% 강함

Usage:
  python scripts/b3_macro_backtest.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Windows cp949 콘솔 회피 — UTF-8 강제
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


CAL_PATH = ROOT / "data" / "macro_events" / "fomc_cpi_calendar.json"
VIX_CACHE = ROOT / "data" / "macro_events" / "vix_daily.csv"
OUT_PATH = ROOT / "data" / "macro_events" / "b3_macro_backtest.json"


def fetch_vix_daily(start: str = "2023-12-01", end: str = "2026-04-26") -> dict[str, float]:
    """yfinance ^VIX 일별 close. CSV 캐시."""
    import yfinance as yf

    if VIX_CACHE.exists():
        # cache hit if newer than 24h
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
    """이벤트일 d → (VIX(d-1), VIX(d)) 반환. 영업일 아닌 경우 직전/당일 영업일 검색."""
    d = datetime.strptime(event_date, "%Y-%m-%d")
    sorted_dates = sorted(vix.keys())

    # find d in sorted_dates (or next available trading day, max 3 days lookahead)
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
    print("=" * 60)
    print("B3 News-IV Macro Backtest — FOMC + CPI vs VIX")
    print("=" * 60)

    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    fomc_events = cal["fomc"]
    cpi_events = cal["cpi"]
    print(f"\n[1] Calendar: {len(fomc_events)} FOMC + {len(cpi_events)} CPI")

    vix = fetch_vix_daily()

    # ── classify each event with rule classifier (sanity check) ──────────
    print("\n[2] Rule classifier sanity check on event titles:")
    n_correct_class = 0
    for ev in fomc_events + cpi_events:
        cls = classify_rule(ev["title"])
        if cls == "macro_shock":
            n_correct_class += 1
    total = len(fomc_events) + len(cpi_events)
    print(f"    classified as macro_shock: {n_correct_class}/{total} "
          f"({n_correct_class/total*100:.1f}%)")

    predicted_shift = IV_SHIFT_RULES["macro_shock"]
    print(f"    IV_SHIFT_RULES['macro_shock'] = {predicted_shift:+.3f} (sign: {'+' if predicted_shift > 0 else '-'})")

    # ── compute ΔVIX for each event ────────────────────────────────────
    print("\n[3] Event-day VIX change measurement")
    rows: list[dict] = []
    for cat, events in [("FOMC", fomc_events), ("CPI", cpi_events)]:
        for ev in events:
            window = find_event_window(ev["date"], vix)
            if window is None:
                continue
            v_prev, v_now = window
            d_abs = v_now - v_prev
            d_pct = d_abs / v_prev * 100.0
            rows.append({
                "category": cat,
                "date": ev["date"],
                "title": ev["title"],
                "vix_prev": round(v_prev, 3),
                "vix_now": round(v_now, 3),
                "delta_abs": round(d_abs, 3),
                "delta_pct": round(d_pct, 2),
                "sign_match": bool(d_abs > 0),  # predicted +0.05 > 0
                "extra": {k: ev.get(k) for k in ("decision", "rate_target", "yoy", "surprise") if k in ev},
            })

    n = len(rows)
    n_match = sum(1 for r in rows if r["sign_match"])
    print(f"    aligned events with VIX data: {n}/{total}")
    print(f"    sign agreement (ΔVIX > 0): {n_match}/{n} = {n_match/n*100:.1f}%")
    print(f"    random baseline: 50.0%, hypothesis threshold: 55.0%")

    # ── per-category breakdown ────────────────────────────────────────
    print("\n[4] Per-category breakdown:")
    for cat in ["FOMC", "CPI"]:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        m = sum(1 for r in cat_rows if r["sign_match"])
        print(f"    {cat:<6s}  n={len(cat_rows):>3d}  match {m/len(cat_rows)*100:.1f}% "
              f"(mean ΔVIX% = {np.mean([r['delta_pct'] for r in cat_rows]):+.2f}%)")

    # ── hawkish-surprise subset ──────────────────────────────────────
    print("\n[5] Hawkish-surprise subset (CPI 'hot' or FOMC hawkish dot plot)")
    hawk_rows = []
    for r in rows:
        ex = r["extra"]
        is_hawk = (ex.get("surprise") == "hot") or ("hawkish" in r["title"].lower())
        if is_hawk:
            hawk_rows.append(r)
    if hawk_rows:
        m = sum(1 for r in hawk_rows if r["sign_match"])
        print(f"    n={len(hawk_rows):>3d}  match {m/len(hawk_rows)*100:.1f}% "
              f"(mean ΔVIX% = {np.mean([r['delta_pct'] for r in hawk_rows]):+.2f}%)")

    dovish_rows = [r for r in rows if r["extra"].get("surprise") == "cool"
                   or r["extra"].get("decision") == "cut"]
    if dovish_rows:
        m = sum(1 for r in dovish_rows if r["sign_match"])
        print(f"    dovish-subset n={len(dovish_rows):>3d}  match {m/len(dovish_rows)*100:.1f}% "
              f"(VIX should DROP → sign would flip; we expect <50% here)")

    # ── magnitude-stratified analysis ─────────────────────────────────
    print("\n[5b] Magnitude-stratified analysis (정직한 진단):")
    abs_pcts = [abs(r["delta_pct"]) for r in rows]
    print(f"    |ΔVIX%| distribution: mean={np.mean(abs_pcts):.2f}%, "
          f"median={np.median(abs_pcts):.2f}%, max={np.max(abs_pcts):.2f}%")
    # large = top quintile by |ΔVIX%|
    threshold = float(np.percentile(abs_pcts, 80))
    large_rows = [r for r in rows if abs(r["delta_pct"]) >= threshold]
    print(f"    Top-20% magnitude (|ΔVIX%| ≥ {threshold:.2f}%): n={len(large_rows)}")
    if large_rows:
        m_large = sum(1 for r in large_rows if r["sign_match"])
        print(f"      sign agreement on LARGE moves: {m_large}/{len(large_rows)} "
              f"= {m_large/len(large_rows)*100:.1f}%")
        large_pos = sum(1 for r in large_rows if r["delta_pct"] > 0)
        print(f"      of those, ΔVIX>0: {large_pos}/{len(large_rows)} "
              f"= {large_pos/len(large_rows)*100:.1f}%")

    # ── 진단: dovish/cool 이벤트는 vol crush (VIX 하락) 이 정상 ──────
    print("\n[5c] Vol-crush hypothesis test (반대 가설):")
    print("    Hypothesis revision: dovish/cool events → IV ↓ (vol crush, not +shift)")
    # 'cool' CPI + 'cut' FOMC 는 unconditional 으로 IV 상승 예측이 틀림
    # → 부호 *반대* 일치율 (즉 ΔVIX < 0) 측정
    soft_rows = [r for r in rows
                 if r["extra"].get("surprise") == "cool"
                 or r["extra"].get("decision") == "cut"]
    if soft_rows:
        m_drop = sum(1 for r in soft_rows if r["delta_pct"] < 0)
        print(f"    soft events n={len(soft_rows)}  ΔVIX<0: {m_drop}/{len(soft_rows)} "
              f"= {m_drop/len(soft_rows)*100:.1f}%  (vol-crush 가설 검증)")

    hot_rows = [r for r in rows
                if r["extra"].get("surprise") == "hot"
                or "hawkish" in r["title"].lower()]
    if hot_rows:
        m_up = sum(1 for r in hot_rows if r["delta_pct"] > 0)
        print(f"    hot/hawkish events n={len(hot_rows)}  ΔVIX>0: "
              f"{m_up}/{len(hot_rows)} = {m_up/len(hot_rows)*100:.1f}%")

    # ── 수정 가설로 다시 평가 ───────────────────────────────────────
    print("\n[5d] Refined classifier (surprise-aware):")
    print("    rule = sign-of-shift conditional on surprise label")
    n_correct = 0
    for r in rows:
        ex = r["extra"]
        if ex.get("surprise") == "hot" or "hawkish" in r["title"].lower():
            predicted_sign = +1
        elif ex.get("surprise") == "cool" or ex.get("decision") == "cut":
            predicted_sign = -1
        else:
            predicted_sign = +1  # default: macro_shock baseline
        actual_sign = 1 if r["delta_pct"] > 0 else -1
        if predicted_sign == actual_sign:
            n_correct += 1
    print(f"    surprise-aware sign match: {n_correct}/{len(rows)} "
          f"= {n_correct/len(rows)*100:.1f}%  (vs naive 34.8%)")

    # ── case studies ─────────────────────────────────────────────────
    print("\n[6] Case studies (manually curated big events):")
    case_rows = []
    for cs in cal["case_studies"]:
        window = find_event_window(cs["date"], vix)
        if window is None:
            continue
        v_prev, v_now = window
        case_rows.append({
            **cs,
            "vix_prev": round(v_prev, 3),
            "vix_now": round(v_now, 3),
            "delta_abs": round(v_now - v_prev, 3),
            "delta_pct": round((v_now - v_prev) / v_prev * 100, 2),
            "rule_class": classify_rule(cs["title"]),
        })
        last = case_rows[-1]
        print(f"    {cs['date']}  {cs['title'][:70]}")
        print(f"      → VIX {last['vix_prev']} → {last['vix_now']} "
              f"({last['delta_pct']:+.2f}%) · rule_class = {last['rule_class']}")

    # ── save ─────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "summary": {
            "total_events": total,
            "rule_classified_as_macro_shock": n_correct_class,
            "aligned_with_vix": n,
            "sign_match": n_match,
            "sign_match_pct": round(n_match / n * 100, 2) if n else 0,
            "surprise_aware_match_pct": round(n_correct / len(rows) * 100, 2) if rows else 0,
            "random_baseline_pct": 50.0,
            "hypothesis_threshold_pct": 55.0,
            "naive_verdict": "PASS" if n and n_match / n > 0.55 else (
                "WEAK" if n and n_match / n > 0.50 else "FAIL"),
            "refined_verdict": "PASS" if rows and n_correct / len(rows) > 0.55 else (
                "WEAK" if rows and n_correct / len(rows) > 0.50 else "FAIL"),
        },
        "by_category": {
            cat: {
                "n": len([r for r in rows if r["category"] == cat]),
                "match_pct": round(
                    sum(1 for r in rows if r["category"] == cat and r["sign_match"])
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
