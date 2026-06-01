"""Phase A4 — production runner 와의 통합 인터페이스.

historical 모델 (data/fair_model_historical.lgbm, Test R²+0.037) 을 받아
stock-level next-hour return 예측 → ETF 비중 가중 합으로 ETF fair direction.

봇 main loop 에서 매 N분마다 maybe_refresh() 호출 → 종목별 예측 갱신.
매 tick 에서 fair_dev(etf_symbol) 로 시그널 받고 strategy.decide_with_fair()
에 넘김.

주의 — 모델 R² +0.037 은 매우 작은 알파. 본격 prod 운영 전에 walk-forward
검증 (다른 기간 데이터로) + paper trading PnL 누적 확인 필수.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

# ETF underlying basket — features ETL 과 동기화 (중복 정의지만 hook 가
# 가벼워야 해서 import 안 함)
_KOSPI200 = {
    "005930": 0.30, "000660": 0.07, "207940": 0.04, "005380": 0.04,
    "035420": 0.03, "000270": 0.02, "005490": 0.02, "035720": 0.02,
    "051910": 0.02, "068270": 0.02,
}
_KOSDAQ150 = {
    "247540": 0.10, "086520": 0.08, "196170": 0.07, "028300": 0.05,
    "091990": 0.04, "035900": 0.03, "263750": 0.03, "058470": 0.03,
    "214150": 0.03, "145020": 0.03,
}
_SEMI = {
    "005930": 0.25, "000660": 0.20, "042700": 0.05, "240810": 0.04,
    "357780": 0.04, "058470": 0.03, "039030": 0.03, "000990": 0.03,
    "036930": 0.03, "005290": 0.02,
}
_BATTERY = {
    "373220": 0.20, "247540": 0.10, "006400": 0.08, "086520": 0.07,
    "003670": 0.05, "066970": 0.04, "096770": 0.04, "121600": 0.03,
    "352820": 0.03, "020150": 0.03,
}
_HEALTHCARE = {
    "068270": 0.18, "207940": 0.15, "128940": 0.06, "000100": 0.05,
    "326030": 0.04, "302440": 0.04, "196170": 0.04, "069620": 0.03,
    "185750": 0.03, "009420": 0.02,
}
ETF_UNDERLYING_HOOK: dict[str, dict[str, float]] = {
    "069500": _KOSPI200, "102110": _KOSPI200, "152100": _KOSPI200,
    "278530": _KOSPI200, "105190": _KOSPI200,
    "229200": _KOSDAQ150, "091160": _SEMI, "305720": _BATTERY,
    "266420": _HEALTHCARE,
}


@dataclass
class FairValuePricer:
    """Historical 모델 wrapper — ETF symbol 별 fair_dev 산출.

    Usage in runner main loop:
        from autotrader.news_signal.hook import FairValuePricer
        pricer = FairValuePricer(
            model_path=ROOT / "data" / "fair_model_historical.lgbm",
            news_dir=ROOT / "data" / "news_cache",
            refresh_min=15,
        )
        # ─ in loop, after fetching minute_bars + news for new bucket:
        pricer.maybe_refresh(now, latest_news_per_stock, latest_bars_per_stock)

        # ─ at decision time:
        for sym in etf_symbols:
            fair_dev = pricer.fair_dev(sym)
            sig, qty, use_inv = strategy.decide_with_fair(
                dev_basket=current_dev, fair_dev=fair_dev,
            )
    """
    model_path: Path
    news_dir: Path
    refresh_min: int = 15
    fair_blend: float = 0.3   # combined_dev = basket*(1-blend) + fair*blend

    _model: object | None = field(default=None, repr=False)
    _last_refresh: datetime | None = field(default=None, repr=False)
    _stock_predictions: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(self.model_path))
        except Exception as e:
            print(f"[FairValuePricer] model load fail: {e}")
            self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def has_predictions(self) -> bool:
        return bool(self._stock_predictions)

    def maybe_refresh(self, now: datetime,
                       news_by_ticker: dict[str, list[dict]],
                       bars_by_ticker: dict) -> bool:
        """N분마다 모든 종목의 최근 뉴스로 1-hour return 예측.

        news_by_ticker: ticker → [{title, description, pub_iso}, ...]  (최근 N분)
        bars_by_ticker: ticker → 5-min bars DataFrame (최근 90분 이상)
        """
        if self._model is None:
            return False
        if (self._last_refresh is not None
                and (now - self._last_refresh).total_seconds() < self.refresh_min * 60):
            return False

        # ai_pricing 분류기 + features 모듈 import (lazy)
        sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))
        from ai_pricing.news_iv.classify import classify_rule
        from autotrader.news_signal.features import sentiment_score, pre_news_features
        from autotrader.news_signal.historical import (
            HIST_FEATURE_COLS, EVENT_CATEGORIES,
        )
        import pandas as pd
        import numpy as np

        # 종목별 가장 최근 뉴스 1건 골라 예측
        new_preds: dict[str, float] = {}
        for ticker, news_list in news_by_ticker.items():
            if not news_list:
                continue
            # 가장 최근 1건
            n = sorted(news_list, key=lambda x: x.get("pub_iso", ""))[-1]
            text = (n.get("title") or "") + " " + (n.get("description") or "")
            event = classify_rule(text)
            sent, n_pos, n_neg = sentiment_score(text)
            try:
                t_news = pd.Timestamp(n.get("pub_iso"))
                if t_news.tzinfo is None:
                    t_news = t_news.tz_localize("Asia/Seoul")
            except Exception:
                continue

            bars = bars_by_ticker.get(ticker)
            if bars is None:
                continue
            pre = pre_news_features(bars, t_news)

            event_id = (EVENT_CATEGORIES.index(event)
                        if event in EVENT_CATEGORIES else len(EVENT_CATEGORIES) - 1)
            ticker_id = abs(hash(ticker)) % 1000  # rough — 학습 시 ticker_id 와 다를 수 있음
            row = {
                "confidence": 1.0, "event_id": float(event_id),
                "ticker_id": float(ticker_id),
                "hour": float(t_news.hour), "minute": float(t_news.minute),
                "weekday": float(t_news.weekday()),
                "minutes_since_open": float((t_news.hour - 9) * 60 + t_news.minute),
                "sent_score": sent, "n_pos_kw": float(n_pos), "n_neg_kw": float(n_neg),
                "momentum_5m": pre.momentum_5m, "momentum_15m": pre.momentum_15m,
                "momentum_30m": pre.momentum_30m, "vol_30m": pre.vol_30m,
                "volume_z": pre.volume_z, "pre_n_bars": float(pre.pre_n_bars),
            }
            X = np.array([[row[c] for c in HIST_FEATURE_COLS]])
            pred = float(self._model.predict(X)[0])
            new_preds[ticker] = pred

        if new_preds:
            self._stock_predictions = new_preds
            self._last_refresh = now
            return True
        return False

    def fair_dev(self, etf_symbol: str) -> float:
        """ETF underlying 종목들의 모델 예측을 비중 가중합 → ETF fair direction.

        Convention: 모델이 +X 예측 = stock 이 X% 오를 거다 = 지금 X% 싸다
                  → fair_dev = -X (negative dev = ETF 가 fair 보다 싸다)
        """
        basket = ETF_UNDERLYING_HOOK.get(etf_symbol)
        if not basket or not self._stock_predictions:
            return 0.0
        weighted = 0.0
        cw = 0.0
        for stock, w in basket.items():
            pred = self._stock_predictions.get(stock)
            if pred is None:
                continue
            weighted += w * pred
            cw += w
        if cw <= 0:
            return 0.0
        avg_pred = weighted / cw
        # negative because positive predicted return = ETF currently undervalued
        return -avg_pred
