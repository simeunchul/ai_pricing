"""Phase A2-D — strengthened features for fair-value model.

이전 학습은 event_id, ticker_id, hour 등 7-feature 만 → R² 0 근처.
근본 원인: 모델이 "뉴스 내용" 자체에 접근 못 함 (event 7-class one-hot 만).

D 패키지 추가 features:
  1) Strengthened sentiment lexicon — 한국어 금융 긍정/부정 키워드 + 강도
     점수 (-1 ~ +1). classify_rule 의 7-event 위에 얹어 더 풍부한 신호.
  2) Pre-news momentum/volume — 뉴스 직전 5/15/30분 가격 변화율,
     거래량 z-score, 변동성. KIS minute bars 에서 추출.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


# --------------------------------------------------------------- sentiment

# 긍정 키워드 (강도 가중)
_POS_KEYWORDS: dict[str, float] = {
    # 강한 긍정 (+1.0)
    "어닝서프라이즈": 1.0, "사상최대": 1.0, "사상 최대": 1.0,
    "신고가": 1.0, "최고가": 0.9, "급등": 0.9, "폭등": 1.0,
    "흑자전환": 1.0, "흑자 전환": 1.0,
    # 중강 긍정 (+0.6)
    "호조": 0.6, "상향": 0.6, "수주": 0.6, "수출 호조": 0.7,
    "강세": 0.5, "신제품": 0.5, "시장점유율 1위": 0.7,
    "기대치 상회": 0.7, "컨센서스 상회": 0.7,
    "최고": 0.5, "신기록": 0.7,
    # 중간 긍정 (+0.4)
    "반등": 0.4, "회복": 0.4, "상승": 0.3, "오름": 0.3,
    "성장": 0.4, "증가": 0.3, "확대": 0.3, "개선": 0.4,
    "투자": 0.2, "협력": 0.3, "체결": 0.3, "추진": 0.2,
    # 분석가 / 등급 (+0.6)
    "매수": 0.6, "buy": 0.6, "강력매수": 0.8,
    "목표가 상향": 0.7, "목표주가 상향": 0.7, "등급 상향": 0.7,
}

# 부정 키워드 (강도 가중, 절댓값)
_NEG_KEYWORDS: dict[str, float] = {
    # 강한 부정 (-1.0)
    "어닝쇼크": 1.0, "급락": 1.0, "폭락": 1.0,
    "적자전환": 1.0, "적자 전환": 1.0, "최저가": 0.9,
    # 중강 부정 (-0.6)
    "부진": 0.6, "하향": 0.6, "감소": 0.4, "감익": 0.7,
    "약세": 0.5, "축소": 0.4, "지연": 0.5,
    "기대치 하회": 0.7, "컨센서스 하회": 0.7,
    "철수": 0.6, "취소": 0.6,
    # 중간 부정 (-0.4)
    "하락": 0.3, "내림": 0.3, "둔화": 0.4, "악화": 0.5,
    "리콜": 0.7, "결함": 0.7, "사고": 0.6,
    "조사": 0.5, "제재": 0.7, "과징금": 0.7, "고소": 0.6, "소송": 0.5,
    "규제": 0.5, "처벌": 0.6, "벌금": 0.7,
    # 분석가 / 등급 (-0.6)
    "매도": 0.6, "sell": 0.6,
    "목표가 하향": 0.7, "목표주가 하향": 0.7, "등급 하향": 0.7,
    # 거시 부정
    "전쟁": 0.8, "분쟁": 0.5, "지정학": 0.4, "긴장": 0.4,
    "금리 인상": 0.4, "금리인상": 0.4,
}


def sentiment_score(text: str) -> tuple[float, int, int]:
    """텍스트의 sentiment score, positive hit count, negative hit count.

    Score = (Σ pos_weights - Σ neg_weights) / max(1, n_words)
        - 긍정 키워드 가중합에서 부정 가중합 빼고 단어 수로 normalize
        - 클램프: [-1.0, +1.0]
    """
    if not text:
        return 0.0, 0, 0
    text_l = text.lower()
    pos_sum = 0.0
    neg_sum = 0.0
    n_pos = 0
    n_neg = 0
    for kw, w in _POS_KEYWORDS.items():
        if kw.lower() in text_l:
            pos_sum += w
            n_pos += 1
    for kw, w in _NEG_KEYWORDS.items():
        if kw.lower() in text_l:
            neg_sum += w
            n_neg += 1
    n_words = max(1, len(re.findall(r"\S+", text_l)))
    raw = (pos_sum - neg_sum) / (n_words ** 0.5)  # √n 로 normalize (긴 글에서 희석 적게)
    return max(-1.0, min(1.0, raw)), n_pos, n_neg


# --------------------------------------------------------------- pre-news momentum

@dataclass
class PreNewsFeatures:
    momentum_5m: float = 0.0
    momentum_15m: float = 0.0
    momentum_30m: float = 0.0
    vol_30m: float = 0.0           # 30분 표준편차 (%)
    volume_z: float = 0.0           # 거래량 z-score (vs 직전 90분 평균)
    pre_n_bars: int = 0             # 사용 가능 bars 수


def pre_news_features(bars: pd.DataFrame, t_news, lookback_bars: int = 18) -> PreNewsFeatures:
    """t_news 직전 5/15/30분 가격/거래량 features.

    bars: 5-min bar DataFrame (index = tz-aware timestamps).
    t_news: pd.Timestamp (tz-aware).
    lookback_bars: 직전 N개 bars 사용 (default 18 = 90분 = vol_z 계산용).

    Bars 가 부족하면 사용 가능한 만큼만 계산하고 0 으로 채움.
    """
    if bars.empty or t_news is None:
        return PreNewsFeatures()

    # t_news 이전 bars
    pre = bars[bars.index < t_news].tail(lookback_bars)
    if pre.empty:
        return PreNewsFeatures()

    closes = pre["Close"].astype(float).values
    volumes = pre["Volume"].astype(float).values
    n = len(closes)

    feats = PreNewsFeatures(pre_n_bars=n)
    last = closes[-1]

    # momentum_5m = 직전 1 bar (5분) 변화율
    if n >= 2 and closes[-2] > 0:
        feats.momentum_5m = (last - closes[-2]) / closes[-2]
    # momentum_15m = 3 bars 전 대비
    if n >= 4 and closes[-4] > 0:
        feats.momentum_15m = (last - closes[-4]) / closes[-4]
    # momentum_30m = 6 bars 전 대비
    if n >= 7 and closes[-7] > 0:
        feats.momentum_30m = (last - closes[-7]) / closes[-7]

    # vol_30m = 직전 6 bars 의 표준편차 (per-bar return)
    if n >= 7:
        recent = closes[-7:]
        rets = (recent[1:] - recent[:-1]) / recent[:-1]
        if len(rets) >= 2:
            mean = rets.mean()
            feats.vol_30m = float(((rets - mean) ** 2).mean() ** 0.5)

    # volume_z = 마지막 bar 거래량의 z-score
    if n >= 6 and volumes[-1] > 0:
        recent_v = volumes[-6:-1]   # 직전 5개 bar 거래량 (마지막 제외)
        if len(recent_v) >= 3:
            mean_v = recent_v.mean()
            std_v = recent_v.std() + 1e-9
            feats.volume_z = float((volumes[-1] - mean_v) / std_v)

    return feats


__all__ = [
    "sentiment_score", "PreNewsFeatures", "pre_news_features",
]
