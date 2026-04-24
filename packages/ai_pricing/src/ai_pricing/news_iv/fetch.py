"""Fetch Korean finance news (Naver RSS + DART disclosures)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class NewsItem:
    source: str
    ticker: str | None
    title: str
    body: str
    url: str
    published: str  # ISO datetime


def fetch_naver_rss(ticker: str | None = None, days: int = 7,
                    cache_dir: str = "data/news_cache") -> list[NewsItem]:
    """Naver 금융 뉴스 RSS. If no ticker, uses general market RSS.

    Requires `feedparser`. Falls back to empty list if not installed.
    """
    try:
        import feedparser
    except ImportError:
        print("[news_iv.fetch] feedparser not installed; returning empty list.")
        return []

    # General market news RSS (Naver finance)
    if ticker is None:
        urls = ["https://finance.naver.com/news/news_list.naver?mode=RSS&section_id=101"]
    else:
        # 종목별 뉴스 - Naver does not expose stable per-ticker RSS publicly; fall back to general
        urls = ["https://finance.naver.com/news/news_list.naver?mode=RSS&section_id=101"]

    cache_key = hashlib.md5(f"{ticker}-{days}".encode()).hexdigest()[:12]
    cache_file = Path(cache_dir) / f"naver_{cache_key}.json"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
        return [NewsItem(**d) for d in json.loads(cache_file.read_text(encoding="utf-8"))]

    items: list[NewsItem] = []
    cutoff = datetime.now() - timedelta(days=days)
    for url in urls:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"[news_iv.fetch] feed error {url}: {e}")
            continue
        for entry in feed.entries:
            published = entry.get("published_parsed")
            if published:
                dt = datetime(*published[:6])
                if dt < cutoff:
                    continue
                pub_iso = dt.isoformat()
            else:
                pub_iso = datetime.now().isoformat()
            items.append(NewsItem(
                source="naver_rss",
                ticker=ticker,
                title=entry.get("title", ""),
                body=entry.get("summary", ""),
                url=entry.get("link", ""),
                published=pub_iso,
            ))

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps([asdict(i) for i in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return items


def fetch_dart_disclosures(corp_code: str, days: int = 7, api_key: str | None = None) -> list[NewsItem]:
    """금융감독원 DART 전자공시 API. Requires API key from opendart.fss.or.kr.

    Returns empty list if API key missing (so training pipelines still run).
    """
    import os

    api_key = api_key or os.environ.get("DART_API_KEY")
    if not api_key:
        print("[news_iv.fetch] DART_API_KEY not set; skipping DART.")
        return []

    try:
        import requests
    except ImportError:
        print("[news_iv.fetch] pip install requests")
        return []

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    url = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bgn_de": start,
        "end_de": end,
        "page_no": 1,
        "page_count": 100,
    }
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    if data.get("status") != "000":
        print(f"[news_iv.fetch] DART status={data.get('status')} {data.get('message')}")
        return []

    items = []
    for row in data.get("list", []):
        items.append(NewsItem(
            source="dart",
            ticker=row.get("stock_code"),
            title=row.get("report_nm", ""),
            body=row.get("rm", ""),
            url=f"http://dart.fss.or.kr/dsaf001/main.do?rcpNo={row.get('rcept_no','')}",
            published=row.get("rcept_dt", ""),
        ))
    return items


__all__ = ["NewsItem", "fetch_naver_rss", "fetch_dart_disclosures"]
