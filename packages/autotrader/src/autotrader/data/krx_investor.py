"""KRX 외국인/기관 일별 순매매 + OHLCV 데이터 페처 (Naver Finance scraper).

원래 plan 은 pykrx 사용 — 그러나 KRX MDC endpoint 가 본 환경에서 403 차단되어
pykrx 의 투자자 데이터 호출도 모두 실패. 대안으로 Naver Finance 의 외국인/기관
거래 페이지(`/item/frgn.naver`)를 페이지네이션 스크래핑.

Naver 페이지 한 장당 약 10영업일, 30 페이지면 약 2.5년 history 확보 가능.
컬럼:
  date, close, volume,
  foreign_net_shares    (외국인 일별 순매수 수량)
  institution_net_shares (기관 일별 순매수 수량)
  foreign_holdings_shares (외국인 누적 보유 수량)
  foreign_ratio_pct       (외국인 보유 비율 %)

flow_ratio = foreign_net_shares / volume  (같은 거래일 → 자연스러운 정규화)
이는 KRX 의 foreign_net_krw / total_value 와 거의 동치 (둘 다 ~VWAP 기준).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[5] / "data" / "krx_cache"

_NAVER_URL = "https://finance.naver.com/item/frgn.naver"
_NAVER_OHLC_URL = "https://finance.naver.com/item/sise_day.naver"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://finance.naver.com/",
}

# 한 페이지 ≈ 10 거래일 → 2년치는 ~50 페이지면 충분
_DEFAULT_MAX_PAGES = 60

# 행 한 줄에서 9개 td 를 순서대로 추출 (date, close, change_abs, change_pct,
# volume, foreign_net, institution_net, foreign_holdings, foreign_ratio).
_NUM = r'[+\-,\d.%]+'
_ROW_RE = re.compile(
    r'<tr[^>]*onMouseOver[^>]*>.*?</tr>',
    re.DOTALL,
)
_DATE_RE = re.compile(r'(\d{4}\.\d{2}\.\d{2})')
_TD_NUM_RE = re.compile(
    r'<td[^>]*class="num"[^>]*>(?P<inner>.*?)</td>',
    re.DOTALL,
)


def _parse_int(s: str) -> int | None:
    """'+2,050,701' → 2050701, '−1,234' → -1234."""
    s = s.replace(",", "").replace("+", "").strip()
    if not s or s == "-":
        return 0
    # Naver uses '−' (Unicode minus) in some cells
    s = s.replace("−", "-")
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_pct(s: str) -> float | None:
    s = s.replace("%", "").replace("+", "").replace("−", "-").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_page(html: str) -> list[dict]:
    """Naver frgn 페이지 HTML → row dict list."""
    rows = []
    for row_html in _ROW_RE.findall(html):
        date_m = _DATE_RE.search(row_html)
        if not date_m:
            continue
        date_str = date_m.group(1).replace(".", "-")

        # 모든 num td 의 내부 텍스트 추출
        nums = []
        for td_inner in _TD_NUM_RE.findall(row_html):
            # strip tags
            text = re.sub(r'<[^>]+>', '', td_inner)
            text = text.strip()
            nums.append(text)

        # 기대: [close, change_abs, change_pct, volume, foreign_net, inst_net,
        #        foreign_holdings, foreign_ratio]
        if len(nums) < 8:
            continue
        close = _parse_int(nums[0])
        volume = _parse_int(nums[3])
        foreign_net = _parse_int(nums[4])
        inst_net = _parse_int(nums[5])
        foreign_holdings = _parse_int(nums[6])
        foreign_ratio = _parse_pct(nums[7])

        if close is None or volume is None:
            continue

        rows.append({
            "date": date_str,
            "close": close,
            "volume": volume,
            "foreign_net_shares": foreign_net or 0,
            "institution_net_shares": inst_net or 0,
            "foreign_holdings_shares": foreign_holdings or 0,
            "foreign_ratio_pct": foreign_ratio,
        })
    return rows


def _fetch_page(symbol: str, page: int, sleep: float = 0.3) -> list[dict]:
    params = {"code": symbol, "page": page}
    r = requests.get(_NAVER_URL, params=params, headers=_HEADERS, timeout=10)
    r.encoding = "euc-kr"
    rows = _parse_page(r.text)
    if sleep > 0:
        time.sleep(sleep)
    return rows


def _parse_ohlc_page(html: str) -> list[dict]:
    """Naver sise_day 페이지 → date/open/high/low/close/volume 추출.

    컬럼 순서: 날짜, 종가, 전일비, 시가, 고가, 저가, 거래량
    """
    rows = []
    for row_html in _ROW_RE.findall(html):
        date_m = _DATE_RE.search(row_html)
        if not date_m:
            continue
        date_str = date_m.group(1).replace(".", "-")
        nums = []
        for td_inner in _TD_NUM_RE.findall(row_html):
            text = re.sub(r'<[^>]+>', '', td_inner).strip()
            nums.append(text)
        # 기대: [close, change, open, high, low, volume]
        if len(nums) < 6:
            continue
        close = _parse_int(nums[0])
        opn = _parse_int(nums[2])
        high = _parse_int(nums[3])
        low = _parse_int(nums[4])
        volume = _parse_int(nums[5])
        if None in (close, opn, high, low, volume):
            continue
        rows.append({
            "date": date_str,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
    return rows


def _fetch_ohlc_page(symbol: str, page: int, sleep: float = 0.3) -> list[dict]:
    params = {"code": symbol, "page": page}
    r = requests.get(_NAVER_OHLC_URL, params=params, headers=_HEADERS, timeout=10)
    r.encoding = "euc-kr"
    rows = _parse_ohlc_page(r.text)
    if sleep > 0:
        time.sleep(sleep)
    return rows


def fetch_ohlc(
    symbol: str,
    max_pages: int = _DEFAULT_MAX_PAGES,
    request_sleep: float = 0.3,
) -> pd.DataFrame:
    """Naver sise_day 페이지네이션 OHLC 수집. 외국인 데이터와 동일한 인덱스
    (DatetimeIndex)로 반환.
    """
    all_rows: list[dict] = []
    for page in range(1, max_pages + 1):
        rows = _fetch_ohlc_page(symbol, page, sleep=request_sleep)
        if not rows:
            break
        all_rows.extend(rows)
        if page > 1 and len(rows) < 10:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
    return df


@dataclass
class InvestorFlowCache:
    cache_dir: Path = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}_investor.parquet"

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df.to_parquet(self._path(symbol))


def fetch_investor_flow(
    symbol: str,
    max_pages: int = _DEFAULT_MAX_PAGES,
    use_cache: bool = True,
    cache: InvestorFlowCache | None = None,
    request_sleep: float = 0.3,
) -> pd.DataFrame:
    """심볼의 외국인/기관 일별 순매매 + OHLCV 데이터 반환.

    Args:
        symbol: 6자리 종목코드 (e.g. "005930")
        max_pages: Naver 페이지네이션 깊이. 한 페이지 ~10영업일 → 60페이지 ~2.4년.
        use_cache: True 면 디스크 캐시 우선 사용.
        cache: 커스텀 캐시 인스턴스 (기본: data/krx_cache/)
        request_sleep: 페이지 간 sleep (Naver 서버 부담 회피)

    Returns:
        DataFrame indexed by date (DatetimeIndex, asc), columns:
          close, volume, foreign_net_shares, institution_net_shares,
          foreign_holdings_shares, foreign_ratio_pct
    """
    cache = cache or InvestorFlowCache()
    if use_cache:
        cached = cache.load(symbol)
        if cached is not None and not cached.empty:
            return cached

    all_rows: list[dict] = []
    for page in range(1, max_pages + 1):
        rows = _fetch_page(symbol, page, sleep=request_sleep)
        if not rows:
            break
        # 같은 페이지에 중복이 있을 수 있으니 우선 누적, 마지막에 dedup
        all_rows.extend(rows)
        # 페이지 간 중복(이전 페이지의 마지막 날짜와 같음) 검출 → 페이지 끝
        if page > 1 and len(rows) < 10:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")

    cache.save(symbol, df)
    return df


def fetch_combined(
    symbol: str,
    max_pages: int = _DEFAULT_MAX_PAGES,
    use_cache: bool = True,
    cache: InvestorFlowCache | None = None,
    request_sleep: float = 0.3,
) -> pd.DataFrame:
    """투자자 흐름(close/volume/외국인 순매매) + OHLC(open/high/low) 결합.

    캐시 형식: data/krx_cache/<symbol>_combined.parquet
    """
    cache = cache or InvestorFlowCache()
    combined_path = cache.cache_dir / f"{symbol}_combined.parquet"
    if use_cache and combined_path.exists():
        return pd.read_parquet(combined_path)

    flow = fetch_investor_flow(
        symbol, max_pages=max_pages, use_cache=use_cache,
        cache=cache, request_sleep=request_sleep,
    )
    if flow.empty:
        return flow

    ohlc = fetch_ohlc(symbol, max_pages=max_pages, request_sleep=request_sleep)
    if ohlc.empty:
        return flow

    # join on date — flow has close/volume already, OHLC adds open/high/low
    merged = flow.join(ohlc[["open", "high", "low"]], how="left")
    # forward-fill OHLC if any holes (Naver pages may diverge slightly)
    merged[["open", "high", "low"]] = merged[["open", "high", "low"]].ffill().bfill()
    merged.to_parquet(combined_path)
    return merged


def combined_cache_path(
    symbol: str, cache: InvestorFlowCache | None = None
) -> Path:
    """`{symbol}_combined.parquet` 캐시 경로."""
    cache = cache or InvestorFlowCache()
    return cache.cache_dir / f"{symbol}_combined.parquet"


def combined_last_date(
    symbol: str, cache: InvestorFlowCache | None = None
) -> "pd.Timestamp | None":
    """combined 캐시의 마지막 거래일. 없거나 비면 None."""
    p = combined_cache_path(symbol, cache)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["close"])
        if df.empty:
            return None
        return df.index.max()
    except Exception:
        return None


def ensure_combined(
    symbol: str,
    *,
    max_age_days: int = 3,
    min_rows: int = 61,
    max_pages: int = _DEFAULT_MAX_PAGES,
    cache: InvestorFlowCache | None = None,
    request_sleep: float = 0.3,
    today: "pd.Timestamp | None" = None,
    allow_fetch: bool = True,
) -> tuple[pd.DataFrame, str]:
    """combined 캐시를 신선하게 보장한다.

    디스크 캐시가 충분히 최신(마지막 거래일이 max_age_days 이내)이고 행 수가
    min_rows 이상이면 그대로 로드, 아니면 네이버에서 재수집해 캐시를 갱신한다.

    Args:
        max_age_days: 캐시 마지막 거래일이 오늘 기준 이 일수 이내면 "신선"으로 간주.
                      음수면 항상 재수집(=force).
        min_rows:     이 행 수 미만이면 신선해도 재수집(얇은 캐시 보강).
        max_pages:    재수집 시 네이버 페이지네이션 깊이 (≈10영업일/페이지).
        allow_fetch:  False 면 캐시가 stale/없어도 네트워크 호출을 하지 않고
                      ("stale"/"missing", 빈 df) 반환 — 쿨다운 중 호출용.

    Returns:
        (df, action): action ∈ {"fresh", "refetched", "failed", "stale", "missing"}.
          - fresh     : 디스크 캐시 그대로 사용
          - refetched : 네이버에서 새로 받아 캐시 갱신
          - failed    : 재수집 시도했으나 빈 결과 (df 는 빈 DataFrame)
          - stale     : allow_fetch=False 이고 캐시가 오래됨 (재수집 안 함)
          - missing   : allow_fetch=False 이고 캐시 없음/불충분 (재수집 안 함)
    """
    cache = cache or InvestorFlowCache()
    p = combined_cache_path(symbol, cache)
    had_cache = False
    if p.exists():
        try:
            df = pd.read_parquet(p)
            if not df.empty and len(df) >= min_rows:
                had_cache = True
                last = df.index.max()
                today_ts = today if today is not None else pd.Timestamp.now()
                age_days = (today_ts.normalize() - last.normalize()).days
                if age_days <= max_age_days:
                    return df, "fresh"
        except Exception:
            pass

    if not allow_fetch:
        return pd.DataFrame(), ("stale" if had_cache else "missing")

    df = fetch_combined(
        symbol, max_pages=max_pages, use_cache=False,
        cache=cache, request_sleep=request_sleep,
    )
    if df is None or df.empty:
        return pd.DataFrame(), "failed"
    return df, "refetched"


def add_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """외국인 순매수 비율 등 derived features 추가.

    flow_ratio: foreign_net_shares / volume
      = 그날 거래량 중 외국인 순매수가 차지한 비중
      양수가 클수록 외국인 매수 강세, 음수가 클수록 매도 강세.

    foreign_ratio_pct_chg: 보유비율 일별 변화 (보조 신호)
    """
    out = df.copy()
    import numpy as np
    vol = out["volume"].replace(0, np.nan)
    out["flow_ratio"] = (out["foreign_net_shares"] / vol).astype(float).fillna(0.0)
    out["inst_ratio"] = (out["institution_net_shares"] / vol).astype(float).fillna(0.0)
    if "foreign_ratio_pct" in out.columns:
        out["foreign_ratio_pct_chg"] = out["foreign_ratio_pct"].diff()
    return out


__all__ = [
    "fetch_investor_flow",
    "fetch_ohlc",
    "fetch_combined",
    "ensure_combined",
    "combined_cache_path",
    "combined_last_date",
    "add_flow_features",
    "InvestorFlowCache",
    "DEFAULT_CACHE_DIR",
]
