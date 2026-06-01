"""ETL: ETF tick log + classified news → 5-min bucket feature dataset.

핵심 아이디어 — *ETF의 dev_bps 만 가지고 매매 결정*하던 봇에, *underlying
종목들의 뉴스 신호*를 합쳐 더 풍부한 시그널을 만든다. 5-분 bucket 단위로:

  per (ETF symbol, 5min bucket):
    ─ tick aggregations: dev_bps_mean/last/std, etf_price_last,
                         momentum_5m, vol_5m, n_ticks
    ─ news aggregations: per underlying stock 의 뉴스를
                         classify_rule/finbert 후 score 산출,
                         가중평균 (ETF 비중 가중)
    ─ label: next-bucket ETF return (분류 모델 학습용)

여러 ETF 가 같은 underlying basket 을 공유 (KOSPI 200 트래커 5종) 하므로
종목별 뉴스 score 는 한 번만 계산하고 ETF 비중에 따라 다중 mapping.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from ai_pricing.news_iv.classify import (
    EventType, classify_rule, classify_finbert,
)


# ETF → underlying basket (자동매매 봇 KOSPI200_BASKET_TOP10 등과 일치)
KOSPI200_BASKET_TOP10 = {
    "005930": 0.30, "000660": 0.07, "207940": 0.04, "005380": 0.04,
    "035420": 0.03, "000270": 0.02, "005490": 0.02, "035720": 0.02,
    "051910": 0.02, "068270": 0.02,
}
KOSDAQ150_BASKET_TOP10 = {
    "247540": 0.10, "086520": 0.08, "196170": 0.07, "028300": 0.05,
    "091990": 0.04, "035900": 0.03, "263750": 0.03, "058470": 0.03,
    "214150": 0.03, "145020": 0.03,
}
SEMICONDUCTOR_BASKET_TOP10 = {
    "005930": 0.25, "000660": 0.20, "042700": 0.05, "240810": 0.04,
    "357780": 0.04, "058470": 0.03, "039030": 0.03, "000990": 0.03,
    "036930": 0.03, "005290": 0.02,
}
BATTERY_BASKET_TOP10 = {
    "373220": 0.20, "247540": 0.10, "006400": 0.08, "086520": 0.07,
    "003670": 0.05, "066970": 0.04, "096770": 0.04, "121600": 0.03,
    "352820": 0.03, "020150": 0.03,
}
HEALTHCARE_BASKET_TOP10 = {
    "068270": 0.18, "207940": 0.15, "128940": 0.06, "000100": 0.05,
    "326030": 0.04, "302440": 0.04, "196170": 0.04, "069620": 0.03,
    "185750": 0.03, "009420": 0.02,
}

ETF_UNDERLYING: dict[str, dict[str, float]] = {
    "069500": KOSPI200_BASKET_TOP10,
    "102110": KOSPI200_BASKET_TOP10,
    "152100": KOSPI200_BASKET_TOP10,
    "278530": KOSPI200_BASKET_TOP10,
    "105190": KOSPI200_BASKET_TOP10,
    "229200": KOSDAQ150_BASKET_TOP10,
    "091160": SEMICONDUCTOR_BASKET_TOP10,
    "305720": BATTERY_BASKET_TOP10,
    "266420": HEALTHCARE_BASKET_TOP10,
}

# 이벤트 → 수치적 sentiment score (-1 ~ +1)
# rule classifier (보수적) 기준. finbert 사용 시 별도 매핑 가능.
EVENT_SENTIMENT: dict[str, float] = {
    "earnings_beat":  +1.0,
    "earnings_miss":  -1.0,
    "mna":            +0.7,
    "rating_change":  +0.4,
    "regulatory":     -0.5,
    "macro_shock":    -0.6,
    "neutral":         0.0,
    "other":           0.0,
}


@dataclass
class ETL_Config:
    tick_log_paths: list[Path]                   # ETF tick log JSON
    news_dir: Path                               # data/news_cache
    bucket_min: int = 5
    classifier: str = "rule"                     # rule | finbert
    label_horizon_buckets: int = 1               # 1 = next bucket return


# -------------------------------------------------------------------- helpers

def _parse_pub(s: str) -> datetime | None:
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)  # KST 가정 (Naver는 KST)
        return dt
    except Exception:
        return None


def _floor_to_bucket(dt: datetime, bucket_min: int) -> datetime:
    epoch_min = int(dt.timestamp() // 60)
    floor = (epoch_min // bucket_min) * bucket_min
    return datetime.fromtimestamp(floor * 60).replace(microsecond=0)


# -------------------------------------------------------------------- buckets

def bucket_ticks(records: list[dict], bucket_min: int = 5) -> pd.DataFrame:
    """ETF tick log → bucket-level aggregations.

    Output cols:
      ts_bucket, symbol, etf_price_last, dev_bps_mean, dev_bps_last,
      dev_bps_std, n_ticks, momentum_5m
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"])
    df["ts_bucket"] = df["ts"].dt.floor(f"{bucket_min}min")
    grouped = df.groupby(["symbol", "ts_bucket"], as_index=False).agg(
        etf_price_last=("etf_price", "last"),
        etf_price_first=("etf_price", "first"),
        dev_bps_mean=("dev_bps", "mean"),
        dev_bps_last=("dev_bps", "last"),
        dev_bps_std=("dev_bps", "std"),
        n_ticks=("etf_price", "count"),
    )
    grouped["dev_bps_std"] = grouped["dev_bps_std"].fillna(0)
    grouped["momentum_bucket"] = (
        grouped["etf_price_last"] - grouped["etf_price_first"]
    ) / grouped["etf_price_first"].replace(0, pd.NA)
    grouped["momentum_bucket"] = grouped["momentum_bucket"].fillna(0)
    return grouped.sort_values(["symbol", "ts_bucket"]).reset_index(drop=True)


def bucket_news(news_dir: Path, tickers: Iterable[str],
                bucket_min: int = 5,
                classifier: str = "rule") -> pd.DataFrame:
    """News headlines → bucket-level sentiment per stock.

    Output cols:
      ts_bucket, ticker, news_count, sentiment_mean, sentiment_max_abs,
      n_positive, n_negative
    """
    rows: list[dict] = []
    classify = classify_finbert if classifier == "finbert" else (
        lambda t: (classify_rule(t), 1.0)
    )

    for ticker in tickers:
        files = sorted(news_dir.glob(f"naver_{ticker}_*.json"))
        if not files:
            continue
        items: list[dict] = []
        for f in files:
            try:
                items.extend(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                continue

        for it in items:
            pub = _parse_pub(it.get("pub_iso", ""))
            if pub is None:
                continue
            text = (it.get("title") or "") + " " + (it.get("description") or "")
            event, conf = classify(text.strip())
            score = EVENT_SENTIMENT.get(event, 0.0) * conf
            rows.append({
                "ts_bucket": _floor_to_bucket(pub, bucket_min),
                "ticker": ticker,
                "event": event,
                "score": score,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    grouped = df.groupby(["ticker", "ts_bucket"], as_index=False).agg(
        news_count=("score", "count"),
        sentiment_mean=("score", "mean"),
        sentiment_max_abs=("score", lambda x: max(abs(x.min()), abs(x.max()))),
        n_positive=("score", lambda x: int((x > 0).sum())),
        n_negative=("score", lambda x: int((x < 0).sum())),
    )
    return grouped.sort_values(["ticker", "ts_bucket"]).reset_index(drop=True)


# -------------------------------------------------------------------- merge

def aggregate_news_to_etf(news_buckets: pd.DataFrame,
                           etf_underlying: dict[str, dict[str, float]] = None
                           ) -> pd.DataFrame:
    """종목별 sentiment → ETF 비중 가중 합으로 ETF-level 시그널.

    Output cols:
      ts_bucket, etf_symbol, news_count_total, sentiment_weighted,
      sentiment_max_abs_weighted, covered_weight
    """
    etf_underlying = etf_underlying or ETF_UNDERLYING
    if news_buckets.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for etf_sym, basket in etf_underlying.items():
        sub = news_buckets[news_buckets["ticker"].isin(basket.keys())].copy()
        if sub.empty:
            continue
        sub["weight"] = sub["ticker"].map(basket)
        for ts_bucket, g in sub.groupby("ts_bucket"):
            covered_w = float(g["weight"].sum())
            if covered_w <= 0:
                continue
            rows.append({
                "ts_bucket": ts_bucket,
                "etf_symbol": etf_sym,
                "news_count_total": int(g["news_count"].sum()),
                "sentiment_weighted": float(
                    (g["sentiment_mean"] * g["weight"]).sum() / covered_w
                ),
                "sentiment_max_abs_weighted": float(
                    (g["sentiment_max_abs"] * g["weight"]).sum() / covered_w
                ),
                "covered_weight": covered_w,
            })

    return (pd.DataFrame(rows).sort_values(["etf_symbol", "ts_bucket"])
            .reset_index(drop=True))


# -------------------------------------------------------------------- main ETL

def build_features(cfg: ETL_Config) -> pd.DataFrame:
    """ETF tick log + news → 학습/추론용 feature dataset.

    Steps:
      1. tick log 모두 로드 → 5-min bucket
      2. underlying 종목들 union → 뉴스 분류 → 5-min bucket
      3. ETF 비중으로 뉴스 sentiment ETF-level 환산
      4. tick bucket × news bucket 조인 (ts_bucket, etf_symbol)
      5. label: next-bucket etf return

    Output cols:
      ts_bucket, etf_symbol, etf_price_last, dev_bps_mean, dev_bps_last,
      dev_bps_std, momentum_bucket, n_ticks, news_count_total,
      sentiment_weighted, sentiment_max_abs_weighted, covered_weight,
      label_return_next  ← 학습 타겟
    """
    # 1) tick aggregation
    all_records: list[dict] = []
    for p in cfg.tick_log_paths:
        try:
            all_records.extend(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[ETL] tick log load fail {p}: {e}")

    tick_df = bucket_ticks(all_records, cfg.bucket_min)
    if tick_df.empty:
        return pd.DataFrame()

    # 2) ETF symbol → underlying tickers union
    etf_syms_present = sorted(tick_df["symbol"].unique())
    underlying_union: set[str] = set()
    for esym in etf_syms_present:
        underlying_union.update(ETF_UNDERLYING.get(esym, {}).keys())

    news_df = bucket_news(cfg.news_dir, underlying_union,
                           cfg.bucket_min, cfg.classifier)

    # 3) news → ETF aggregate
    etf_news_df = aggregate_news_to_etf(news_df)

    # 4) join
    merged = tick_df.merge(
        etf_news_df,
        left_on=["symbol", "ts_bucket"],
        right_on=["etf_symbol", "ts_bucket"],
        how="left",
    )
    # tick 만 있고 news 가 없는 bucket → 0 으로 채움
    for col in ("news_count_total", "sentiment_weighted",
                "sentiment_max_abs_weighted", "covered_weight"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    merged = merged.drop(columns=["etf_symbol"], errors="ignore")

    # 5) label = 다음 bucket 의 etf return (per symbol)
    merged = merged.sort_values(["symbol", "ts_bucket"]).reset_index(drop=True)
    merged["etf_price_next"] = merged.groupby("symbol")["etf_price_last"].shift(
        -cfg.label_horizon_buckets
    )
    merged["label_return_next"] = (
        (merged["etf_price_next"] - merged["etf_price_last"])
        / merged["etf_price_last"].replace(0, pd.NA)
    )
    merged = merged.dropna(subset=["label_return_next"]).reset_index(drop=True)
    merged = merged.drop(columns=["etf_price_next"])
    return merged
