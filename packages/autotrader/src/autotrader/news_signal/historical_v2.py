"""Phase A2-D / C — 6개월 1h-bar + DART 공시 events 로 robust 학습 dataset.

기존 v1 (5분봉 + Naver 30일) 한계:
  - 시간 granularity 짧고, 데이터 30일 → walk-forward 거의 noise

v2 changes:
  - 1h interval (yfinance 730d 가능, 안전하게 180d 사용)
  - Event source 다중화: DART 공시 + Naver 뉴스 둘 다
  - Pre-event features: 1h, 3h, 6h, 1d momentum + 24h volatility
  - Label: next 1h ETF return (= 다음 시간봉 close 변화)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))


# ─── 18 종목 KIS code ↔ DART corp_code 매핑 (방금 확보)
TICKER_TO_CORP_CODE: dict[str, str] = {
    "005930": "00126380",  # 삼성전자
    "000660": "00164779",  # SK하이닉스
    "005380": "00137997",  # 현대차
    "035420": "00266961",  # NAVER
    "035720": "00258801",  # 카카오
    "207940": "00877059",  # 삼성바이오로직스
    "051910": "00356361",  # LG화학
    "068270": "00413046",  # 셀트리온
    "005490": "00155319",  # POSCO홀딩스
    "066570": "00401731",  # LG전자
    "000270": "00106641",  # 기아
    "373220": "01515323",  # LG에너지솔루션
    "247540": "01160363",  # 에코프로비엠
    "086520": "00536541",  # 에코프로
    "006400": "00126362",  # 삼성SDI
    "042700": "00161383",  # 한미반도체
    "128940": "00828497",  # 한미약품
    "000100": "00145109",  # 유한양행
}


# ─── DART 보고서명 → 이벤트 type 매핑 (rough)
DART_EVENT_TYPES: dict[str, str] = {
    "사업보고서": "earnings_disclosure",
    "분기보고서": "earnings_disclosure",
    "반기보고서": "earnings_disclosure",
    "주요사항보고서": "material_event",
    "공시정정": "amendment",
    "최대주주변경": "ownership_change",
    "타법인주식및출자증권취득결정": "investment",
    "주식양수도": "ownership_change",
    "단일판매·공급계약체결": "contract",
    "유형자산취득결정": "capex",
    "신규시설투자": "capex",
    "감자결정": "capital_restructure",
    "유상증자결정": "capital_raise",
    "전환사채권발행결정": "capital_raise",
    "회사합병결정": "mna",
    "회사분할결정": "mna",
    "주식분할결정": "stock_split",
    "주식배당결정": "dividend",
    "현금배당결정": "dividend",
    "임원ㆍ주요주주특정증권등소유상황보고서": "insider",
    "주식등의대량보유상황보고서": "ownership_change",
    "기업설명회(IR)개최": "ir",
    "수시공시": "general",
    "조회공시요구(현저한시황변동)": "price_query",
}


def _classify_dart_report(report_nm: str) -> str:
    """DART report_nm → event type. coarse keyword 매칭."""
    for kw, ev in DART_EVENT_TYPES.items():
        if kw in report_nm:
            return ev
    return "other"


def fetch_dart_events(corp_code: str, days: int = 180,
                       page_count: int = 100) -> list[dict]:
    """단일 corp_code 6개월치 공시 list (page 여러 개)."""
    import dart_fetch as dart
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    import requests
    items_all: list[dict] = []
    for page_no in range(1, 11):
        try:
            r = requests.get(
                "https://opendart.fss.or.kr/api/list.json",
                params={
                    "crtfc_key": dart.get_api_key(),
                    "corp_code": corp_code,
                    "bgn_de": start, "end_de": end,
                    "page_no": page_no, "page_count": page_count,
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "000":
                break
            page_items = data.get("list", [])
            if not page_items:
                break
            items_all.extend(page_items)
            total_pages = data.get("total_page", 1)
            if page_no >= total_pages:
                break
        except Exception as e:
            print(f"  [DART] {corp_code} page {page_no} fail: {e}")
            break
    return items_all


def _parse_dart_ts(rcept_dt: str, rcept_no: str) -> pd.Timestamp:
    """DART rcept_dt = 'YYYYMMDD' + rcept_no 마지막 자리 = 접수번호.
    실제 시각 정확치 알기 어려움 — 9:00 AM KST 로 근사 (장개시 + 익일 영향).
    """
    try:
        d = datetime.strptime(rcept_dt, "%Y%m%d")
        # 시각 미상 → 9:00 KST 로 근사
        return pd.Timestamp(d.replace(hour=9, minute=0)).tz_localize("Asia/Seoul")
    except Exception:
        return pd.NaT


# ─────────────────────────────────────────────────── 1h-bar features

def pre_event_features_1h(bars: pd.DataFrame, t_event: pd.Timestamp) -> dict:
    """1h-bar 기반 pre-event features.

    bars index 가 tz-aware 라고 가정. t_event 도 tz-aware.
    """
    if bars.empty or pd.isna(t_event):
        return {k: 0.0 for k in (
            "momentum_1h", "momentum_3h", "momentum_6h", "momentum_1d",
            "vol_24h", "volume_z_1h", "pre_n_bars",
        )}

    pre = bars[bars.index < t_event].tail(48)  # 직전 ~48시간
    if pre.empty:
        return {k: 0.0 for k in (
            "momentum_1h", "momentum_3h", "momentum_6h", "momentum_1d",
            "vol_24h", "volume_z_1h", "pre_n_bars",
        )}

    closes = pre["Close"].astype(float).values
    volumes = pre["Volume"].astype(float).values
    n = len(closes)
    last = closes[-1]

    feats = {
        "momentum_1h": 0.0, "momentum_3h": 0.0, "momentum_6h": 0.0,
        "momentum_1d": 0.0, "vol_24h": 0.0, "volume_z_1h": 0.0,
        "pre_n_bars": float(n),
    }
    # 1h, 3h, 6h, 1d (=24bars 가정)
    for label, lookback in [("momentum_1h", 1), ("momentum_3h", 3),
                              ("momentum_6h", 6), ("momentum_1d", 24)]:
        if n >= lookback + 1 and closes[-lookback - 1] > 0:
            feats[label] = (last - closes[-lookback - 1]) / closes[-lookback - 1]

    # vol_24h
    if n >= 25:
        recent = closes[-25:]
        rets = (recent[1:] - recent[:-1]) / recent[:-1]
        feats["vol_24h"] = float(rets.std())

    # volume z (vs 24h 평균)
    if n >= 24 and volumes[-1] > 0:
        recent_v = volumes[-25:-1]
        if len(recent_v) >= 5:
            mean_v = recent_v.mean()
            std_v = recent_v.std() + 1e-9
            feats["volume_z_1h"] = float((volumes[-1] - mean_v) / std_v)

    return feats


# ───────────────────────────────────────────────── full ETL

@dataclass
class ETLv2Config:
    tickers: list[str]
    days_history: int = 180
    bars_period: str = "180d"
    bars_interval: str = "1h"
    label_horizon: int = 1   # 다음 N개 1h-bar 후 close 기준 return


FEATURE_COLS_V2 = [
    "ticker_id", "hour", "weekday",
    "event_id", "is_dart",
    "momentum_1h", "momentum_3h", "momentum_6h", "momentum_1d",
    "vol_24h", "volume_z_1h", "pre_n_bars",
]
LABEL_COL_V2 = "label_next_h_return"


# DART event 카테고리 → numeric
DART_EVENT_CATEGORIES = sorted(set(DART_EVENT_TYPES.values()) | {"other", "neutral"})


def build_dataset_v2(cfg: ETLv2Config) -> pd.DataFrame:
    """6개월 1h-bar + DART 공시 → 학습 dataset.

    각 row = (ticker, t_event, pre_features, label, source).
    source: 'dart' (= 공시) 또는 'random' (= 비-event baseline 추가 가능)
    """
    sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
    from autotrader.data.minute_bars import fetch_minute_bars, get_bar_at_or_after

    rows: list[dict] = []
    print(f"=== ETL v2: {len(cfg.tickers)} tickers, {cfg.days_history}d ===")

    for ticker in cfg.tickers:
        print(f"  [{ticker}] fetching 1h bars + DART...")
        # 1) 1h-bar fetch
        try:
            bars = fetch_minute_bars(
                ticker, period=cfg.bars_period, interval=cfg.bars_interval,
            )
        except Exception as e:
            print(f"    bars fetch fail: {e}")
            continue
        if bars.empty:
            print(f"    bars empty, skip")
            continue
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("Asia/Seoul")
        else:
            bars.index = bars.index.tz_convert("Asia/Seoul")

        # 2) DART events
        corp_code = TICKER_TO_CORP_CODE.get(ticker)
        if not corp_code:
            print(f"    no DART corp_code mapping")
            continue
        events = fetch_dart_events(corp_code, days=cfg.days_history)
        print(f"    DART events: {len(events)}")

        # 3) 각 공시 row 화
        for ev in events:
            t_event = _parse_dart_ts(ev.get("rcept_dt"), ev.get("rcept_no"))
            if pd.isna(t_event):
                continue
            # 다음 1h-bar 가 있어야 label 산출 가능
            entry_bar = get_bar_at_or_after(bars, t_event.to_pydatetime())
            if entry_bar is None:
                continue
            label_idx = bars.index.get_loc(entry_bar.name) + cfg.label_horizon
            if label_idx >= len(bars):
                continue
            entry_p = float(entry_bar["Close"])
            exit_bar = bars.iloc[label_idx]
            exit_p = float(exit_bar["Close"])
            if entry_p <= 0 or exit_p <= 0:
                continue
            label = (exit_p - entry_p) / entry_p

            event_type = _classify_dart_report(ev.get("report_nm", ""))
            pre = pre_event_features_1h(bars, t_event)

            rows.append({
                "ticker": ticker,
                "t_event": t_event,
                "source": "dart",
                "event": event_type,
                "is_dart": 1.0,
                "report_nm": ev.get("report_nm", "")[:80],
                "label_next_h_return": label,
                **pre,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # numeric encodings
    df["ticker_id"] = pd.Categorical(df["ticker"]).codes.astype(float)
    df["hour"] = df["t_event"].apply(lambda t: float(t.hour))
    df["weekday"] = df["t_event"].apply(lambda t: float(t.weekday()))
    df["event_id"] = df["event"].apply(
        lambda e: float(DART_EVENT_CATEGORIES.index(e))
        if e in DART_EVENT_CATEGORIES else float(len(DART_EVENT_CATEGORIES) - 1)
    )
    return df


__all__ = [
    "TICKER_TO_CORP_CODE", "DART_EVENT_TYPES",
    "fetch_dart_events", "build_dataset_v2",
    "ETLv2Config", "FEATURE_COLS_V2", "LABEL_COL_V2",
    "pre_event_features_1h",
]
