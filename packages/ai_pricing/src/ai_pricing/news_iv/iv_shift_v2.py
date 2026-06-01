"""B3 v2 — surprise-aware ΔIV with magnitude weighting.

v1 진단 (docs/2026-04-26/B3_news_iv_backtest.html):
  - 단방향 ΔIV (macro_shock = +0.05) → naive 부호 일치율 34.8% (50% 미만 FAIL)
  - 매크로 발표일은 평균 vol crush (불확실성 해소 → IV ↓) 가 정상
  - magnitude top-20% 만 60% 일치 — 시그널은 surprise 강도에만 존재

v2 가설:
  ΔIV(event, surprise_label, confidence) =
      base_shift[event_type]
      × surprise_direction(surprise_label)   // +1 hawkish/hot, -1 dovish/cool/expected
      × confidence                           // 0~1 from FinBERT/Claude
      × magnitude_modifier                   // hot/hawkish 강도

  여기서:
    surprise_direction("hot")     = +1.0   // VIX ↑ 예상
    surprise_direction("hawkish") = +1.0
    surprise_direction("cool")    = -0.6   // VIX ↓ 예상 (vol crush, 약한 magnitude)
    surprise_direction("cut")     = -0.6   // 예상된 cut → vol crush
    surprise_direction("inline")  = -0.3   // 예상치 일치 → 약한 vol crush
    surprise_direction("hold")    = -0.3
    surprise_direction(None)      =  0.0
"""

from __future__ import annotations

from typing import Literal

from ai_pricing.news_iv.classify import EventType
from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES as IV_SHIFT_RULES_V1


SurpriseLabel = Literal["hot", "cool", "inline", "hawkish", "dovish", "cut", "hike", "hold"]


SURPRISE_DIRECTION: dict[str, float] = {
    "hot":     +1.0,
    "hawkish": +1.0,
    "hike":    +0.5,    # 예상된 hike 는 약한 vol up
    "cool":    -0.6,
    "dovish":  -0.6,
    "cut":     -0.6,
    "inline":  -0.3,
    "hold":    -0.3,
}


def adjust_iv_v2(
    base_iv: float,
    event: EventType,
    confidence: float = 1.0,
    surprise: str | None = None,
    title: str | None = None,
) -> float:
    """surprise-aware bidirectional adjustment with magnitude weighting.

    base_shift × surprise_direction × confidence
    """
    base_shift = IV_SHIFT_RULES_V1.get(event, 0.0)
    direction = SURPRISE_DIRECTION.get(surprise, 0.0) if surprise else 0.0

    if title and "hawkish" in title.lower():
        direction = +1.0
    elif title and "dovish" in title.lower():
        direction = -0.6

    if surprise is None and direction == 0.0:
        direction = +1.0

    shift = abs(base_shift) * direction * confidence
    adjusted = base_iv + shift
    return max(0.01, min(adjusted, 2.0))


def predict_sign_v2(
    event: EventType,
    surprise: str | None = None,
    title: str | None = None,
    decision: str | None = None,
) -> int:
    """예측 부호: +1 (IV ↑) / -1 (IV ↓) / 0 (모름)."""
    if title:
        t = title.lower()
        if "hawkish" in t:
            return +1
        if "dovish" in t:
            return -1

    if surprise == "hot":
        return +1
    if surprise == "cool":
        return -1
    if surprise == "inline":
        return -1
    if decision == "cut":
        return -1
    if decision == "hike":
        return +1
    if decision == "hold":
        return -1

    if event == "macro_shock":
        return -1
    if event == "earnings_miss" or event == "regulatory":
        return +1
    if event == "earnings_beat":
        return -1

    return 0


__all__ = ["adjust_iv_v2", "predict_sign_v2", "SURPRISE_DIRECTION"]
