"""B3 → B4 NewsAwareHedger wrapper (#2 통합).

기본 Buehler / BSM Δ 정책 위에 B3 의 macro shock 신호로 hedge 보수성 보강.

Logic:
  1. base_action = base_policy(obs)
  2. recent_news → classify_rule → event
  3. if event ∈ HIGH_VOL_EVENTS:
       action = clip(base_action + buffer, action_low, action_high)
     else:
       action = base_action

Buffer 크기는 event magnitude (IV_SHIFT_RULES) 비례.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# B3 import
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from ai_pricing.news_iv.classify import classify_rule, EventType
from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES, dominant_event


HIGH_VOL_EVENTS: set[str] = {
    "earnings_miss", "macro_shock", "regulatory", "mna",
}


@dataclass
class HedgeBufferRule:
    """Map event → hedge buffer (additive). Positive = more hedge."""
    base_buffer: float = 0.05      # 5% extra hedge for any HIGH_VOL_EVENT
    magnitude_scale: float = 2.0   # buffer ∝ |ΔIV| × scale


def event_to_buffer(event: EventType, confidence: float = 1.0,
                     rule: HedgeBufferRule | None = None) -> float:
    """이벤트 → hedge buffer."""
    rule = rule or HedgeBufferRule()
    if event not in HIGH_VOL_EVENTS:
        return 0.0
    delta_iv = abs(IV_SHIFT_RULES.get(event, 0.0))
    return rule.base_buffer + rule.magnitude_scale * delta_iv * confidence


class NewsAwareHedger:
    """Wrapper around any base hedger.

    Args
    ----
    base_predict : callable(obs) -> action ∈ [low, high]
        Buehler policy.predict, or BSMDeltaHedger.act, etc.
    action_low, action_high : float
        Action box.
    rule : HedgeBufferRule
    """

    def __init__(self, base_predict, action_low: float = 0.0,
                  action_high: float = 1.0, rule: HedgeBufferRule | None = None):
        self.base_predict = base_predict
        self.action_low = action_low
        self.action_high = action_high
        self.rule = rule or HedgeBufferRule()
        self._last_event: EventType = "neutral"
        self._last_buffer: float = 0.0

    def update_news(self, news_texts: list[str]) -> tuple[EventType, float]:
        """Reclassify recent news. Call this every market tick / news refresh."""
        events = [(classify_rule(t), 1.0) for t in news_texts]
        ev, conf = dominant_event(events)
        self._last_event = ev
        self._last_buffer = event_to_buffer(ev, conf, self.rule)
        return ev, conf

    def act(self, obs) -> float:
        """obs → adjusted hedge."""
        base = float(self.base_predict(obs))
        adjusted = base + self._last_buffer
        return float(np.clip(adjusted, self.action_low, self.action_high))

    @property
    def last_event(self) -> EventType:
        return self._last_event

    @property
    def last_buffer(self) -> float:
        return self._last_buffer


__all__ = ["NewsAwareHedger", "HedgeBufferRule", "event_to_buffer", "HIGH_VOL_EVENTS"]
