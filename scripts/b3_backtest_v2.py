"""B3 News-IV 백테스트 v2 — yfinance ticker.news + 일별 historical IV.

DART/네이버 RSS 의 한국 시장 IV 데이터 부재 문제 우회.
야후 의 ticker.news 로 영어 헤드라인 받고, 동일 ticker 의 옵션 chain 으로
일별 ATM IV 추정 → 부호 일치율 측정.

Universe: 변동성 큰 6 종목 (AAPL, MSFT, NVDA, GOOGL, META, TSLA)
Method:
  1. 각 ticker.news → list of (title, providerPublishTime)
  2. classify_rule (영어 키워드 보강됨) → event
  3. publish_date 의 ATM IV 와 다음 영업일 ATM IV 차이 측정
     (옵션 chain 의 atm IV 평균 사용)
  4. predicted ΔIV 부호 vs actual ΔIV 부호 일치율
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

from ai_pricing.news_iv.classify import classify_rule
from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES


def collect_news(tickers: list[str]) -> list[dict]:
    import yfinance as yf
    rows = []
    for sym in tickers:
        try:
            news = yf.Ticker(sym).news or []
        except Exception:
            news = []
        for n in news:
            content = n.get("content") or {}
            title = content.get("title") or n.get("title") or ""
            pub = content.get("pubDate") or n.get("pubDate")
            if not (title and pub):
                continue
            rows.append({"ticker": sym, "title": title, "pub": pub})
        time.sleep(0.1)
    return rows


def parse_pub(s: str):
    try:
        # "2026-04-22T17:46:31Z" or unix int
        if isinstance(s, (int, float)):
            return datetime.utcfromtimestamp(int(s))
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def get_atm_iv(ticker: str, asof: datetime) -> float | None:
    """Approximate ATM IV: nearest-expiry ATM call IV (today only — no history)."""
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        spot = tk.history(period="1d")["Close"].iloc[-1]
        exps = tk.options[:3]
        ivs = []
        for e in exps:
            ch = tk.option_chain(e).calls
            if len(ch) == 0:
                continue
            atm = ch.iloc[(ch["strike"] - spot).abs().argsort().iloc[:3]]
            iv = atm["impliedVolatility"].mean()
            if not np.isnan(iv):
                ivs.append(float(iv))
        return float(np.mean(ivs)) if ivs else None
    except Exception:
        return None


def get_realized_short_term_move(ticker: str, asof: datetime, days: int = 2) -> float | None:
    """주가 절대 변동률 (단기 realized vol proxy)."""
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        # asof 기준 ±5 영업일 데이터
        start = asof - timedelta(days=10)
        end = asof + timedelta(days=10)
        hist = tk.history(start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"))
        if len(hist) < 4:
            return None
        # asof 이후 N 영업일 절대 수익률
        idx = hist.index.searchsorted(asof.replace(tzinfo=None))
        if idx + days >= len(hist):
            return None
        p_now = hist["Close"].iloc[idx]
        p_then = hist["Close"].iloc[idx + days]
        return float(abs(np.log(p_then / p_now)))
    except Exception:
        return None


def main():
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN"]
    print(f"=== B3 Backtest v2 — yfinance news, {len(tickers)} tickers ===\n")

    print("[1] News collection...")
    rows = collect_news(tickers)
    print(f"    {len(rows)} headlines")

    if not rows:
        print("    yfinance news returned empty (rate limit?)"); return

    print("\n[2] Event classification (rule-based, English+Korean)...")
    events = []
    for r in rows:
        ev = classify_rule(r["title"])
        if ev != "neutral":
            events.append({**r, "event": ev,
                           "predicted_delta_iv": IV_SHIFT_RULES[ev]})
    print(f"    non-neutral events: {len(events)}")
    if not events:
        print("    Sample headlines (no rule match):")
        for r in rows[:5]:
            print(f"     · [{r['ticker']}] {r['title'][:80]}")
        return

    ev_dist = {}
    for e in events:
        ev_dist[e["event"]] = ev_dist.get(e["event"], 0) + 1
    print(f"    distribution: {ev_dist}")

    # Show top examples
    print(f"\n    Sample classified events:")
    for e in events[:8]:
        print(f"     · [{e['ticker']}] {e['event']:<14s}  {e['title'][:70]}")

    print(f"\n[3] Realized move (proxy for ΔIV) — fetching historical data...")
    enriched = []
    for e in events[:30]:    # cap for speed
        pub_dt = parse_pub(e["pub"])
        if pub_dt is None:
            continue
        if (datetime.utcnow() - pub_dt).days > 60:
            continue
        rm = get_realized_short_term_move(e["ticker"], pub_dt, days=2)
        if rm is None:
            continue
        enriched.append({**e, "pub_dt": pub_dt.strftime("%Y-%m-%d"),
                         "realized_2d_abs_return": rm})

    n = len(enriched)
    print(f"    enriched: {n}")
    if n == 0:
        print("    No enrichment possible"); return

    # Hypothesis: |predicted_delta_iv| 이 큰 event 일수록 realized 절대 수익률 큼
    # (즉 IV shift 가설이 맞다면 큰 충격 이벤트 후 실제 주가 변동도 큼)
    big_thr = 0.03   # |ΔIV| > 3 vp 인 강한 이벤트
    big = [x for x in enriched if abs(x["predicted_delta_iv"]) >= big_thr]
    small = [x for x in enriched if abs(x["predicted_delta_iv"]) < big_thr]

    if big and small:
        big_mean = np.mean([x["realized_2d_abs_return"] for x in big])
        small_mean = np.mean([x["realized_2d_abs_return"] for x in small])
        print(f"\n[4] Realized 2-day |return| 비교:")
        print(f"    강한 event (|ΔIV|≥3vp,  n={len(big):>3d}): mean abs return {big_mean*100:.2f}%")
        print(f"    약한 event (|ΔIV|<3vp,  n={len(small):>3d}): mean abs return {small_mean*100:.2f}%")
        ratio = big_mean / max(small_mean, 1e-6)
        print(f"    ratio (big/small) = {ratio:.2f}× — 가설은 >1 예상")
        verdict = "SUPPORTS" if ratio > 1.2 else ("WEAK" if ratio > 1.0 else "REJECTS")
        print(f"    Verdict: {verdict}")
    else:
        print(f"\n[4] Insufficient samples to test hypothesis (big={len(big)}, small={len(small)})")

    out = ROOT / "data" / "b3_backtest_v2.json"
    out.write_text(json.dumps({
        "n_news_total": len(rows),
        "n_events_classified": len(events),
        "n_enriched": n,
        "ev_distribution": ev_dist,
        "events_sample": [
            {**{k: v for k, v in e.items() if k != "pub"}}
            for e in enriched[:20]
        ],
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
