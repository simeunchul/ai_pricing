"""News → T+5분 진입 / T+1시간 청산 단기 매매 backtest.

Long-only (KIS 모의투자 공매도 불가). 각 뉴스 헤드라인 분류 후 양의 이벤트만
매수, 일정 시간 후 청산. 거래비용 0.27% (수수료 0.045% × 2 + 거래세 0.23%) 차감.

목표:
  Layer-B3 의 일별 backtest 가 모두 fail 한 후 분봉 단위로 재검증.
  사용자 가설 = "T+5분 진입 / T+1시간 청산 단순 momentum 이 작동하는가".
  retail PEAD 기준 1시간은 짧지만 (보통 1일~60일 drift) 거래비용 대비 짧은
  윈도우의 마진 검증 자체가 학습 가치.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

# B3 분류기 재사용
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from ai_pricing.news_iv.classify import (
    EventType, classify_rule, classify_finbert,
)
from autotrader.data.minute_bars import (
    fetch_minute_bars, get_bar_at_or_after,
)
from autotrader.news_signal.features import (
    sentiment_score, pre_news_features,
)


# Long-only 매매 가능한 이벤트 (KIS 모의투자 공매도 불가).
POSITIVE_EVENTS: set[EventType] = {"earnings_beat", "mna", "rating_change"}
# 매매 불가 (시그널은 기록).
NEGATIVE_EVENTS: set[EventType] = {"earnings_miss", "regulatory", "macro_shock"}


@dataclass
class BacktestConfig:
    tickers: list[str]
    news_dir: Path
    classifier: str = "rule"            # "rule" | "finbert"
    entry_delay_min: int = 5            # T+5분 진입
    hold_min: int = 60                  # T+5분 후 60분 보유
    cost_bps: float = 27.0              # round-trip 거래비용 (basis points)
    market_open_hhmm: tuple[int, int] = (9, 0)
    market_close_hhmm: tuple[int, int] = (15, 30)
    tradable_events: set[EventType] | None = None  # None = POSITIVE_EVENTS


def _classify_one(text: str, backend: str) -> tuple[EventType, float]:
    if backend == "finbert":
        return classify_finbert(text)
    return classify_rule(text), 1.0


def _load_news_for_ticker(ticker: str, news_dir: Path) -> list[dict]:
    """data/news_cache/naver_<ticker>_<DATE>.json 로드. 여러 날짜 파일 union."""
    files = sorted(news_dir.glob(f"naver_{ticker}_*.json"))
    items: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items.extend(data)
        except Exception:
            continue
    return items


def _parse_pub(s: str) -> datetime | None:
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _is_market_hours(t: datetime, cfg: BacktestConfig) -> bool:
    h, m = t.hour, t.minute
    open_h, open_m = cfg.market_open_hhmm
    close_h, close_m = cfg.market_close_hhmm
    after_open = (h, m) >= (open_h, open_m)
    before_close = (h, m) <= (close_h, close_m)
    return after_open and before_close and t.weekday() < 5


def run_news_intraday_backtest(cfg: BacktestConfig) -> pd.DataFrame:
    """뉴스 헤드라인별 backtest. 각 헤드라인 → 한 trade.

    Returns DataFrame columns:
      ticker, event, confidence, t_news, t_entry, t_exit,
      entry, exit, gross_pnl, net_pnl, title, tradable
    """
    tradable_events = cfg.tradable_events or POSITIVE_EVENTS
    trades: list[dict] = []

    for ticker in cfg.tickers:
        news = _load_news_for_ticker(ticker, cfg.news_dir)
        if not news:
            continue

        bars = fetch_minute_bars(ticker, period="60d", interval="5m")
        if bars.empty:
            continue

        for n in news:
            text = f"{n.get('title','')} {n.get('description','')}"
            event, conf = _classify_one(text, cfg.classifier)

            # 시계열 시점 정렬
            t_news = _parse_pub(n.get("pub_iso", ""))
            if t_news is None:
                continue

            # 장중 발생 뉴스만 (장외 발생은 다음날 시가 영향 — 별도 분석 필요)
            t_news_kst = pd.Timestamp(t_news).tz_localize("Asia/Seoul") \
                if t_news.tzinfo is None else pd.Timestamp(t_news).tz_convert("Asia/Seoul")
            if not _is_market_hours(t_news_kst.to_pydatetime(), cfg):
                continue

            t_entry = t_news_kst + pd.Timedelta(minutes=cfg.entry_delay_min)
            t_exit = t_news_kst + pd.Timedelta(minutes=cfg.entry_delay_min + cfg.hold_min)

            entry_bar = get_bar_at_or_after(bars, t_entry.to_pydatetime())
            exit_bar = get_bar_at_or_after(bars, t_exit.to_pydatetime())
            if entry_bar is None or exit_bar is None:
                continue
            if entry_bar.name >= exit_bar.name:
                continue
            # 동일 거래일 only (overnight 회피)
            if entry_bar.name.date() != exit_bar.name.date():
                continue

            entry_p = float(entry_bar["Close"])
            exit_p = float(exit_bar["Close"])
            if entry_p <= 0 or exit_p <= 0:
                continue

            gross = (exit_p - entry_p) / entry_p
            cost = cfg.cost_bps / 10_000.0
            net = gross - cost

            # === Phase A2-D 강화 features ===
            # 1) 강화 sentiment lexicon
            full_text = f"{n.get('title','')} {n.get('description','')}"
            sent_score, n_pos_kw, n_neg_kw = sentiment_score(full_text)
            # 2) pre-news momentum/volume (5/15/30분 전)
            pre = pre_news_features(bars, t_news_kst, lookback_bars=18)

            trades.append({
                "ticker": ticker,
                "event": event,
                "confidence": conf,
                "t_news": t_news_kst.isoformat(),
                "t_entry": entry_bar.name.isoformat(),
                "t_exit": exit_bar.name.isoformat(),
                "entry": entry_p,
                "exit": exit_p,
                "gross_pnl": gross,
                "net_pnl": net,
                "title": (n.get("title") or "")[:120],
                "tradable": event in tradable_events,
                # ─ 강화 features
                "sent_score": sent_score,
                "n_pos_kw": n_pos_kw,
                "n_neg_kw": n_neg_kw,
                "momentum_5m": pre.momentum_5m,
                "momentum_15m": pre.momentum_15m,
                "momentum_30m": pre.momentum_30m,
                "vol_30m": pre.vol_30m,
                "volume_z": pre.volume_z,
                "pre_n_bars": pre.pre_n_bars,
            })

    return pd.DataFrame(trades)


def summarize_trades(df: pd.DataFrame, only_tradable: bool = True) -> dict:
    """Aggregate stats: hit rate, mean P&L, Sharpe, per-event breakdown."""
    if df.empty:
        return {"n": 0, "note": "no trades"}

    sub = df[df["tradable"]] if only_tradable else df
    n = len(sub)
    if n == 0:
        return {"n": 0, "note": "no tradable trades"}

    hit = float((sub["net_pnl"] > 0).mean())
    mean_pnl = float(sub["net_pnl"].mean())
    std_pnl = float(sub["net_pnl"].std())
    # Annualize: trades/day → assume 5 trades/day per ticker, 252 trading days
    # Conservatively use per-trade Sharpe × sqrt(N) with N≈sample size
    if std_pnl > 0:
        sharpe_pertrade = mean_pnl / std_pnl
        # Rough annualization assuming ~50 trades/year per ticker
        ann_factor = (252 ** 0.5)
        sharpe_ann = sharpe_pertrade * ann_factor
    else:
        sharpe_ann = float("nan")

    by_event = (
        sub.groupby("event")
           .agg(n=("net_pnl", "count"),
                hit=("net_pnl", lambda s: (s > 0).mean()),
                mean=("net_pnl", "mean"),
                std=("net_pnl", "std"))
           .reset_index()
    )

    by_ticker = (
        sub.groupby("ticker")
           .agg(n=("net_pnl", "count"),
                hit=("net_pnl", lambda s: (s > 0).mean()),
                mean=("net_pnl", "mean"))
           .reset_index()
    )

    return {
        "n": n,
        "hit_rate": hit,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe_per_trade": mean_pnl / std_pnl if std_pnl > 0 else float("nan"),
        "sharpe_annualized": sharpe_ann,
        "total_pnl_sum": float(sub["net_pnl"].sum()),
        "best_trade": float(sub["net_pnl"].max()),
        "worst_trade": float(sub["net_pnl"].min()),
        "by_event": by_event.to_dict("records"),
        "by_ticker": by_ticker.to_dict("records"),
    }


__all__ = [
    "BacktestConfig",
    "run_news_intraday_backtest",
    "summarize_trades",
    "POSITIVE_EVENTS",
    "NEGATIVE_EVENTS",
]
