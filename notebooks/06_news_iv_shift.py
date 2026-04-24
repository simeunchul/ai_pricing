"""Notebook W6 — News / LLM → IV shift."""

# %%
from pricing.bsm import BSMInputs
from ai_pricing.news_iv.fetch import NewsItem
from ai_pricing.news_iv.pipeline import price_with_news

# %% Synthetic news (so this runs without network)
news = [
    NewsItem(source="test", ticker="005930", title="삼성전자 2분기 어닝쇼크 실적 부진",
             body="반도체 수요 감소로 실적 부진...", url="", published="2026-04-20T09:00:00"),
    NewsItem(source="test", ticker=None, title="한국은행 기준금리 인상 발표",
             body="물가 안정 위한 금리 인상...", url="", published="2026-04-21T10:00:00"),
]

option = BSMInputs(S=100, K=100, T=0.5, r=0.03, q=0, sigma=0.22)
res = price_with_news(option, ticker="005930", news=news, classifier="rule")

for k, v in res.items():
    print(f"{k}: {v}")
