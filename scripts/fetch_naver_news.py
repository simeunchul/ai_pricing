"""한국 뉴스 수집 — Naver Developers Search API.

B3 한국 시장 백테스트용. 종목별로 최근 N일 뉴스 헤드라인 수집.

Usage:
  python scripts/fetch_naver_news.py
  python scripts/fetch_naver_news.py --tickers 005930,000660,005380 --days 30

Output:
  data/news_cache/naver_<ticker>_<YYYYMMDD>.json
  Each: list of {"title", "description", "link", "pub_iso", "ticker"}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# 종목 코드 → 검색 keyword map. 검색은 회사명이 효과 좋음.
DEFAULT_TICKERS: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "035420": "NAVER",
    "035720": "카카오",
    "207940": "삼성바이오로직스",
    "051910": "LG화학",
    "068270": "셀트리온",
    "005490": "POSCO홀딩스",
    "066570": "LG전자",
}


def load_env(env_path: Path = ROOT / ".env") -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def naver_news_search(query: str, display: int = 100,
                       sort: str = "date", start: int = 1) -> dict:
    cid = os.environ.get("NAVER_CLIENT_ID")
    csec = os.environ.get("NAVER_CLIENT_SECRET")
    if not (cid and csec):
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수 또는 .env 필요")

    r = requests.get(
        "https://openapi.naver.com/v1/search/news.json",
        params={"query": query, "display": display, "sort": sort, "start": start},
        headers={"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def clean_html(s: str) -> str:
    """Naver 응답의 <b>...</b>, &quot; 등 제거."""
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&quot;", '"').replace("&amp;", "&")
           .replace("&lt;", "<").replace("&gt;", ">")
           .replace("&apos;", "'").replace("&nbsp;", " "))
    return s.strip()


def parse_pub(rfc822: str) -> datetime | None:
    try:
        return parsedate_to_datetime(rfc822).replace(tzinfo=None)
    except Exception:
        return None


def fetch_ticker_news(ticker: str, query: str, days: int = 30,
                       max_pages: int = 10, sleep_sec: float = 0.1) -> list[dict]:
    """종목 1개에 대해 sort=date 으로 최근 N일치 수집.

    Naver 검색 API: start 1~1000, display 1~100. 한 query 당 최대 1000건.
    sort=date 라서 최신순. 우리는 N일 cutoff 만나면 멈춤.
    """
    cutoff = datetime.now() - timedelta(days=days)
    items_out: list[dict] = []
    seen_links = set()

    for page in range(1, max_pages + 1):
        start_idx = (page - 1) * 100 + 1
        if start_idx > 1000:
            break
        try:
            res = naver_news_search(query, display=100, sort="date", start=start_idx)
        except Exception as e:
            print(f"  [{ticker}] page {page} fail: {e}")
            break
        items = res.get("items", [])
        if not items:
            break

        stop = False
        for it in items:
            pub = parse_pub(it.get("pubDate", ""))
            if pub is None:
                continue
            if pub < cutoff:
                stop = True
                break
            link = it.get("originallink") or it.get("link", "")
            if link in seen_links:
                continue
            seen_links.add(link)
            items_out.append({
                "ticker": ticker,
                "query": query,
                "title": clean_html(it.get("title", "")),
                "description": clean_html(it.get("description", "")),
                "link": link,
                "pub_iso": pub.isoformat(),
            })
        if stop:
            break
        time.sleep(sleep_sec)

    return items_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS.keys()),
                    help="comma-separated 종목코드 (기본: 시총 상위 10)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out-dir", default="data/news_cache")
    args = ap.parse_args()

    load_env()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    print(f"=== Naver News Fetch ({len(tickers)} tickers, {args.days}d) ===\n")
    total = 0
    for ticker in tickers:
        query = DEFAULT_TICKERS.get(ticker, ticker)
        print(f"[{ticker}] {query}", end=" ... ")
        items = fetch_ticker_news(ticker, query, days=args.days)
        out = out_dir / f"naver_{ticker}_{today}.json"
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        total += len(items)
        print(f"{len(items)} items → {out.name}")

    print(f"\n→ total: {total} headlines across {len(tickers)} tickers")
    print(f"→ cache: {out_dir}")


if __name__ == "__main__":
    main()
