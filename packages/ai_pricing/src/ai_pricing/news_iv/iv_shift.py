"""Event → ΔIV lookup.

Initial values from Araci (2019) FinBERT paper + Lopez de Prado empirical ranges.
These are expert priors; Layer C backtest can re-calibrate.
"""

from __future__ import annotations

from ai_pricing.news_iv.classify import EventType

# ΔIV in absolute vol points (e.g. +0.03 = +3 vol points)
IV_SHIFT_RULES: dict[EventType, float] = {
    "earnings_miss":  +0.03,
    "earnings_beat":  -0.01,
    "regulatory":     +0.02,
    "macro_shock":    +0.05,
    "mna":            +0.04,
    "rating_change":  +0.01,
    "neutral":         0.00,
}


def adjust_iv(base_iv: float, event: EventType, confidence: float = 1.0) -> float:
    """Linear scale by confidence, clip to realistic vol range."""
    shift = IV_SHIFT_RULES.get(event, 0.0) * confidence
    adjusted = base_iv + shift
    return max(0.01, min(adjusted, 2.0))


def dominant_event(events: list[tuple[EventType, float]]) -> tuple[EventType, float]:
    """Pick the event with highest |shift| * confidence magnitude."""
    if not events:
        return "neutral", 0.0
    scored = [(ev, conf, abs(IV_SHIFT_RULES[ev]) * conf) for ev, conf in events]
    scored.sort(key=lambda x: -x[2])
    ev, conf, _ = scored[0]
    return ev, conf


__all__ = ["IV_SHIFT_RULES", "adjust_iv", "dominant_event"]
