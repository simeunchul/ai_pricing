"""End-to-end: news → classify → IV shift → re-price."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Literal

from pricing.bsm import BSMInputs, call_price, put_price

from ai_pricing.news_iv.classify import (
    EventType,
    classify_claude,
    classify_finbert,
    classify_rule,
)
from ai_pricing.news_iv.fetch import NewsItem, fetch_naver_rss
from ai_pricing.news_iv.iv_shift import adjust_iv, dominant_event


ClassifierName = Literal["rule", "finbert", "claude"]


def _classify(text: str, backend: ClassifierName) -> tuple[EventType, float]:
    if backend == "claude":
        return classify_claude(text)
    if backend == "finbert":
        return classify_finbert(text)
    return classify_rule(text), 1.0


def price_with_news(
    option: BSMInputs,
    ticker: str | None = None,
    news: list[NewsItem] | None = None,
    as_of: datetime | None = None,
    classifier: ClassifierName = "rule",
    opt_type: str = "call",
) -> dict:
    """Return dict with base_price, adjusted_price, iv_shift, events."""
    as_of = as_of or datetime.now()

    if news is None:
        news = fetch_naver_rss(ticker=ticker, days=3)

    events: list[tuple[EventType, float]] = []
    for n in news:
        text = f"{n.title} {n.body}"
        events.append(_classify(text, classifier))

    ev, conf = dominant_event(events)
    iv_adj = adjust_iv(option.sigma, ev, conf)

    pricer = call_price if opt_type == "call" else put_price
    base_price = pricer(option)
    adjusted_option = replace(option, sigma=iv_adj)
    adjusted_price = pricer(adjusted_option)

    return {
        "base_price": base_price,
        "adjusted_price": adjusted_price,
        "base_iv": option.sigma,
        "adjusted_iv": iv_adj,
        "iv_shift": iv_adj - option.sigma,
        "dominant_event": ev,
        "dominant_confidence": conf,
        "all_events": events,
        "n_news": len(news),
        "as_of": as_of.isoformat(),
    }


__all__ = ["price_with_news", "ClassifierName"]
