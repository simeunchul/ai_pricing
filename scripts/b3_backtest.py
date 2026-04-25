"""B3 News-IV 실데이터 백테스트.

질문: "뉴스 이벤트 분류 + IV_SHIFT_RULES 가 다음날 실제 IV 변화 부호를 맞추는가?"

Pipeline:
  1. DART 최근 N일 한화 / KOSPI 관련 공시 list
  2. (옵션) 네이버 RSS 같은 기간 헤드라인
  3. 각 헤드라인 → classify_rule 으로 event 분류 → ΔIV 예측 (부호)
  4. 이벤트 발생일 dᵢ → SPY IV 의 dᵢ → dᵢ+1 변화 측정 (yfinance 일별 IV 근사)
     * 한국 시장 IV (VKOSPI 등) 못 쓰니까 SPY 로 대체 데모
  5. 부호 일치율 측정. 50% 가 무의미 (랜덤), >55% 가 가설 검증.

Usage:
  python scripts/b3_backtest.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))

from ai_pricing.news_iv.classify import classify_rule
from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES


def fetch_dart_recent_disclosures(days: int = 30, limit: int = 60) -> list[dict]:
    """List recent disclosures across the entire market for richer event sample.
    한화 단일은 거의 매일 ELS 발행 공시뿐이라 multi-corp 로 확장."""
    import os, sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from dart_fetch import get_api_key

    import requests
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    # 시장 전체 공시 (corp_code 없음, 회차 1)
    url = "https://opendart.fss.or.kr/api/list.json"
    r = requests.get(url, params={
        "crtfc_key": get_api_key(),
        "bgn_de": start, "end_de": end,
        "page_no": 1, "page_count": limit,
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "000":
        print(f"[dart] {data.get('status')} {data.get('message')}")
        return []
    return data.get("list", [])[:limit]


def fetch_iv_history_spy(days_back: int = 35) -> list[dict]:
    """Use yfinance to grab SPY's 30-day IV proxy via VIX-like measure.
    For simplicity: use option-chain ATM IV at each tradable date."""
    import yfinance as yf
    vix = yf.Ticker("^VIX").history(period=f"{days_back}d")
    return [{"date": d.strftime("%Y-%m-%d"), "vix_close": float(c)}
            for d, c in zip(vix.index.date, vix["Close"])]


def main():
    print("=== B3 News-IV Backtest ===\n")

    # 1. DART events
    print("[1] DART 시장 공시 30일 리스트...")
    discs = fetch_dart_recent_disclosures(days=30, limit=80)
    print(f"    {len(discs)} disclosures")

    # 2. Classify each
    rows = []
    for d in discs:
        title = d.get("report_nm", "")
        body = d.get("rm", "")
        text = f"{title} {body}"
        ev = classify_rule(text)
        if ev != "neutral":
            rows.append({
                "date": d.get("rcept_dt", ""),
                "corp": d.get("corp_name", ""),
                "title": title,
                "event": ev,
                "delta_iv_predicted": IV_SHIFT_RULES[ev],
            })
    n_events = len(rows)
    print(f"[2] non-neutral 분류: {n_events}건")
    if rows:
        ev_dist = {}
        for r in rows:
            ev_dist[r["event"]] = ev_dist.get(r["event"], 0) + 1
        print(f"    이벤트 분포: {ev_dist}")

    # 3. IV history (VIX 일별 종가 — SPY 의 IV proxy)
    print(f"\n[3] VIX (S&P500 IV proxy) 일별 종가 fetching...")
    iv_hist = fetch_iv_history_spy(days_back=35)
    iv_by_date = {h["date"]: h["vix_close"] for h in iv_hist}
    if not iv_by_date:
        print("    VIX 데이터 못 받음 (네트워크?)"); return
    sample_dates = list(iv_by_date.keys())[:3]
    print(f"    {len(iv_by_date)}일치, 샘플 날짜: {sample_dates}")

    # 4. Sign agreement
    aligned = []
    for r in rows:
        d = r["date"]
        if not d or len(d) != 8:
            continue
        d_iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        # find d_iso and the next available date
        all_dates = sorted(iv_by_date.keys())
        if d_iso not in all_dates:
            continue
        idx = all_dates.index(d_iso)
        if idx + 1 >= len(all_dates):
            continue
        d_next = all_dates[idx + 1]
        iv_today = iv_by_date[d_iso]
        iv_next = iv_by_date[d_next]
        actual_delta = iv_next - iv_today
        predicted_delta = r["delta_iv_predicted"] * 100   # rules in vol points; VIX in %, comparable
        sign_match = (np.sign(actual_delta) == np.sign(predicted_delta))
        aligned.append({**r, "iv_today": iv_today, "iv_next": iv_next,
                        "actual_delta_vp": actual_delta,
                        "predicted_delta_vp": predicted_delta,
                        "sign_match": bool(sign_match)})

    n_aligned = len(aligned)
    n_match = sum(1 for a in aligned if a["sign_match"])
    print(f"\n[4] 정렬 가능 events: {n_aligned}/{n_events}")
    if n_aligned:
        agree = n_match / n_aligned
        print(f"    Sign agreement: {n_match}/{n_aligned} = {agree*100:.1f}%")
        print(f"    Random baseline: 50.0%, 가설 검증선: 55.0%")
        verdict = "PASS" if agree > 0.55 else ("WEAK" if agree > 0.5 else "FAIL")
        print(f"    Verdict: {verdict}")

    # 5. Per-event breakdown
    print(f"\n[5] 이벤트별 sign match:")
    by_event = {}
    for a in aligned:
        by_event.setdefault(a["event"], []).append(a["sign_match"])
    for ev, matches in sorted(by_event.items()):
        rate = sum(matches) / len(matches) if matches else 0
        print(f"    {ev:<18s} n={len(matches):>3d}  match {rate*100:.1f}%")

    # Save
    out = ROOT / "data" / "b3_backtest_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_disclosures_total": len(discs),
        "n_events_classified": n_events,
        "n_aligned_with_iv": n_aligned,
        "sign_agreement_rate": n_match / n_aligned if n_aligned else 0,
        "events": aligned[:30],   # cap
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
