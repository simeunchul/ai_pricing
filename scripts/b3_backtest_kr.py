"""B3 한국 시장 백테스트 — Naver 뉴스 + KODEX 200 ETF.

가설: "큰 macro/event 뉴스 → 다음 영업일 KODEX 200 의 절대 수익률 ↑"
     (즉 뉴스가 vol 충격을 예측한다는 가설)

V-KOSPI 200 직접 못 받음 (KRX 인증 차단) → KODEX 200 일별 수익률 절대값을
realized vol proxy 로 사용.

Pipeline:
  1. data/news_cache/naver_*.json 로드 (9000+ headlines)
  2. classify_rule (한글 키워드) → event 분류
  3. 이벤트일 D → KODEX 200 의 |return(D+1)| vs |return(평균)| 비교
  4. magnitude top-K event vs random baseline 부호/크기 일치율 측정

Usage:
  python scripts/b3_backtest_kr.py
  python scripts/b3_backtest_kr.py --top-pct 0.20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from ai_pricing.news_iv.classify import classify_rule
from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES


def load_news(cache_dir: Path) -> list[dict]:
    items = []
    for p in sorted(cache_dir.glob("naver_*.json")):
        items.extend(json.loads(p.read_text(encoding="utf-8")))
    return items


def fetch_kodex_history(days_back: int = 90) -> dict[str, float]:
    """KODEX 200 일별 종가. {YYYY-MM-DD: close}."""
    from pykrx import stock
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv(start, end, "069500")
    return {d.strftime("%Y-%m-%d"): float(c)
            for d, c in zip(df.index.date, df["종가"])}


def compute_returns(prices: dict[str, float]) -> dict[str, float]:
    """일별 log return. {YYYY-MM-DD: log(P_t / P_{t-1})}."""
    sorted_dates = sorted(prices.keys())
    rets = {}
    for i in range(1, len(sorted_dates)):
        d_prev, d = sorted_dates[i - 1], sorted_dates[i]
        rets[d] = float(np.log(prices[d] / prices[d_prev]))
    return rets


def find_next_trading_day(date_str: str, sorted_dates: list[str]) -> str | None:
    """date_str 이후 첫 영업일."""
    for d in sorted_dates:
        if d > date_str:
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/news_cache")
    ap.add_argument("--top-pct", type=float, default=0.20,
                    help="magnitude top X% events 만 보는 strict 모드")
    args = ap.parse_args()

    cache_dir = ROOT / args.cache
    print(f"=== B3 한국 시장 백테스트 ===\n")

    # 1. Load news
    print(f"[1] Loading news cache...")
    news = load_news(cache_dir)
    print(f"    total {len(news)} headlines, {len(set(n['ticker'] for n in news))} tickers")

    # 2. Classify
    print(f"\n[2] Classify (rule, 한글 키워드)...")
    classified = []
    for n in news:
        # title + description 합쳐서 분류 (recall 높임)
        text = f"{n['title']} {n['description']}"
        ev = classify_rule(text)
        if ev != "neutral":
            classified.append({
                **n,
                "event": ev,
                "predicted_delta_iv": IV_SHIFT_RULES[ev],
                "pub_date": n["pub_iso"][:10],
            })
    n_cls = len(classified)
    print(f"    non-neutral: {n_cls}")

    if n_cls == 0:
        print("    분류 결과 0건 — 키워드 매칭 한계")
        return

    dist = Counter(c["event"] for c in classified)
    print(f"    distribution: {dict(dist)}")

    # 3. KODEX 200 history
    print(f"\n[3] KODEX 200 일별 가격 fetching...")
    prices = fetch_kodex_history(days_back=90)
    rets = compute_returns(prices)
    sorted_dates = sorted(rets.keys())
    print(f"    {len(rets)} 영업일 (최근 90 calendar days)")

    # baseline: 전체 평균 |return|
    all_abs_rets = [abs(r) for r in rets.values()]
    baseline_mean_abs = float(np.mean(all_abs_rets))
    baseline_median_abs = float(np.median(all_abs_rets))
    print(f"    baseline mean |return|: {baseline_mean_abs*100:.3f}%")
    print(f"    baseline median |return|: {baseline_median_abs*100:.3f}%")

    # 4. Align events to next trading day
    print(f"\n[4] Aligning events → next trading day return...")
    aligned = []
    for c in classified:
        d_event = c["pub_date"]
        # 같은 날 또는 다음 영업일의 return 측정
        d_target = d_event if d_event in rets else find_next_trading_day(d_event, sorted_dates)
        if d_target is None:
            continue
        ret = rets[d_target]
        aligned.append({
            **c,
            "target_date": d_target,
            "actual_return": ret,
            "actual_abs_return": abs(ret),
            "predicted_delta_iv_abs": abs(c["predicted_delta_iv"]),
        })
    n_aligned = len(aligned)
    print(f"    aligned: {n_aligned}/{n_cls}")

    if n_aligned < 10:
        print("    too few aligned — skip stats"); return

    # 5. Hypothesis tests
    print(f"\n[5] Hypothesis tests")

    # (a) sign agreement (매우 약한 가설 — 부정 event 면 다음날 음수 return?)
    # earnings_miss/regulatory/macro/mna/rating 중 negative 라벨 매핑
    NEGATIVE_EVENTS = {"earnings_miss", "regulatory", "macro_shock", "mna", "rating_change"}
    n_neg_ev = sum(1 for a in aligned if a["event"] in NEGATIVE_EVENTS)
    n_neg_match = sum(1 for a in aligned
                       if a["event"] in NEGATIVE_EVENTS and a["actual_return"] < 0)
    if n_neg_ev > 0:
        sign_rate = n_neg_match / n_neg_ev
        print(f"  (a) negative event → next-day return < 0 일치율: "
              f"{n_neg_match}/{n_neg_ev} = {sign_rate*100:.1f}%")
        verdict = "PASS >55%" if sign_rate > 0.55 else ("WEAK >50%" if sign_rate > 0.5 else "FAIL")
        print(f"      verdict: {verdict}")

    # (b) magnitude — event 일 다음날 |return| 이 baseline 보다 큰가?
    event_abs = [a["actual_abs_return"] for a in aligned]
    event_abs_mean = float(np.mean(event_abs))
    print(f"  (b) event 다음날 mean |return|: {event_abs_mean*100:.3f}%  "
          f"(baseline {baseline_mean_abs*100:.3f}%)")
    ratio = event_abs_mean / baseline_mean_abs
    print(f"      ratio: {ratio:.3f}× — 가설은 >1 예상")
    verdict_b = "PASS >1.2" if ratio > 1.2 else ("WEAK >1.0" if ratio > 1.0 else "FAIL")
    print(f"      verdict: {verdict_b}")

    # (c) top magnitude events (predicted ΔIV 가 큰 event 만)
    aligned_sorted = sorted(aligned, key=lambda a: -a["predicted_delta_iv_abs"])
    n_top = max(10, int(len(aligned) * args.top_pct))
    top = aligned_sorted[:n_top]
    top_abs_mean = float(np.mean([a["actual_abs_return"] for a in top]))
    top_ratio = top_abs_mean / baseline_mean_abs
    print(f"  (c) top {args.top_pct*100:.0f}% magnitude (n={n_top}) "
          f"mean |return|: {top_abs_mean*100:.3f}%")
    print(f"      ratio: {top_ratio:.3f}× — 가설은 >1.5 예상 (강한 event 만)")
    verdict_c = "PASS >1.5" if top_ratio > 1.5 else ("WEAK >1.2" if top_ratio > 1.2 else "FAIL")
    print(f"      verdict: {verdict_c}")

    # (d) per-event-type breakdown
    print(f"\n[6] Per-event-type")
    by_event: dict = {}
    for a in aligned:
        by_event.setdefault(a["event"], []).append(a["actual_abs_return"])
    print(f"    {'event':<18s}{'n':>5s}{'mean |ret|':>14s}{'ratio':>10s}")
    for ev, vals in sorted(by_event.items(), key=lambda x: -len(x[1])):
        m = float(np.mean(vals))
        rt = m / baseline_mean_abs
        print(f"    {ev:<18s}{len(vals):>5d}{m*100:>12.3f}%  {rt:>8.2f}×")

    # 7. Save
    out = ROOT / "data" / "b3_backtest_kr.json"
    out.write_text(json.dumps({
        "n_news_total": len(news),
        "n_classified": n_cls,
        "n_aligned": n_aligned,
        "event_distribution": dict(dist),
        "baseline_mean_abs_return": baseline_mean_abs,
        "event_mean_abs_return": event_abs_mean,
        "ratio_event_vs_baseline": ratio,
        "top_pct": args.top_pct,
        "top_n": n_top,
        "top_ratio": top_ratio,
        "negative_event_sign_rate": (n_neg_match / n_neg_ev) if n_neg_ev > 0 else None,
        "events_sample": aligned[:20],
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
