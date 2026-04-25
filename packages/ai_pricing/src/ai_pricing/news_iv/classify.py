"""Event classification for news headlines.

Two backends:
  - FinBERT (snunlp/KR-FinBert-SC) — free, offline after first download
  - Claude API — more accurate zero-shot; requires ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import os
from typing import Literal

EventType = Literal[
    "earnings_miss",
    "earnings_beat",
    "regulatory",
    "macro_shock",
    "mna",
    "rating_change",
    "neutral",
]

EVENT_TYPES: tuple[EventType, ...] = (
    "earnings_miss", "earnings_beat", "regulatory",
    "macro_shock", "mna", "rating_change", "neutral",
)


# -----------------------------------------------------------------------------
# Rule-based (no model needed, useful fallback & unit tests)
# -----------------------------------------------------------------------------

_KEYWORD_RULES: dict[EventType, tuple[str, ...]] = {
    "earnings_miss":  (
        "어닝쇼크", "실적 부진", "적자 전환", "하향", "earnings miss",
        "missed estimates", "disappointing earnings", "earnings shortfall",
        "lowered guidance", "profit warning", "loss widens",
    ),
    "earnings_beat":  (
        "어닝서프라이즈", "실적 호조", "흑자 전환", "상향", "earnings beat",
        "beat estimates", "exceeded expectations", "raised guidance",
        "record profit", "strong earnings",
    ),
    "regulatory":     (
        "규제", "제재", "과징금", "조사", "시정", "regulatory",
        "antitrust", "investigation", "fine", "lawsuit", "sec",
        "violation", "probe",
    ),
    "macro_shock":    (
        "금리 인상", "금리 인하", "지정학", "전쟁", "유가 급등", "환율 급변",
        "rate hike", "rate cut", "fed", "fomc", "inflation surge",
        "oil spike", "war", "geopolitical",
    ),
    "mna":            (
        "인수", "합병", "M&A", "주식교환", "지분 인수",
        "acquires", "acquisition", "merger", "buyout", "takeover",
        "stake purchase",
    ),
    "rating_change":  (
        "등급 하향", "등급 상향", "신용등급", "rating",
        "downgrade", "upgrade", "moody's", "s&p", "fitch",
    ),
}


def classify_rule(text: str) -> EventType:
    text_l = text.lower()
    for ev, kws in _KEYWORD_RULES.items():
        if any(kw.lower() in text_l for kw in kws):
            return ev
    return "neutral"


# -----------------------------------------------------------------------------
# FinBERT (Korean)
# -----------------------------------------------------------------------------


class _FinBertSingleton:
    tokenizer = None
    model = None

    @classmethod
    def get(cls):
        if cls.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            name = "snunlp/KR-FinBert-SC"
            cls.tokenizer = AutoTokenizer.from_pretrained(name)
            cls.model = AutoModelForSequenceClassification.from_pretrained(name)
            cls.model.eval()
        return cls.tokenizer, cls.model


def classify_finbert(text: str) -> tuple[EventType, float]:
    """Returns (event, confidence). Maps FinBERT sentiment to coarse event class."""
    try:
        import torch
    except ImportError:
        return classify_rule(text), 0.5

    try:
        tok, mdl = _FinBertSingleton.get()
    except Exception as e:
        print(f"[news_iv.classify] FinBERT load failed: {e}; using rule-based.")
        return classify_rule(text), 0.5

    with torch.no_grad():
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
        logits = mdl(**inputs).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).numpy()

    # FinBERT labels: [negative, neutral, positive] (snunlp model convention)
    neg, neu, pos = probs
    rule = classify_rule(text)
    if rule != "neutral":
        # keyword match takes priority but scale confidence by sentiment strength
        conf = float(max(neg, pos))
        return rule, conf
    if pos > 0.6:
        return "earnings_beat", float(pos)
    if neg > 0.6:
        return "earnings_miss", float(neg)
    return "neutral", float(neu)


# -----------------------------------------------------------------------------
# Claude API
# -----------------------------------------------------------------------------


_CLAUDE_PROMPT = """You classify Korean/English financial news headlines into one of these event types:

- earnings_miss: 실적 부진/어닝쇼크
- earnings_beat: 실적 호조/서프라이즈
- regulatory: 규제/제재/과징금
- macro_shock: 거시 충격(금리, 지정학, 유가)
- mna: M&A, 인수합병
- rating_change: 신용/투자 등급 변경
- neutral: 그 외, 또는 가격 영향 불명확

Return ONLY a JSON object like:
{"event": "earnings_miss", "confidence": 0.85}

News:
"""


def classify_claude(text: str, model: str = "claude-haiku-4-5-20251001") -> tuple[EventType, float]:
    """Use Anthropic API (Haiku for cost). Falls back to rule-based if no API key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return classify_rule(text), 0.5

    try:
        from anthropic import Anthropic
    except ImportError:
        print("[news_iv.classify] pip install anthropic; using rule-based fallback.")
        return classify_rule(text), 0.5

    client = Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": _CLAUDE_PROMPT + text}],
        )
        raw = resp.content[0].text  # type: ignore[attr-defined]
        # try strict JSON parse
        start = raw.find("{")
        end = raw.rfind("}") + 1
        obj = json.loads(raw[start:end])
        ev = obj.get("event", "neutral")
        conf = float(obj.get("confidence", 0.5))
        if ev not in EVENT_TYPES:
            ev = "neutral"
        return ev, conf  # type: ignore[return-value]
    except Exception as e:
        print(f"[news_iv.classify] Claude error: {e}; rule fallback.")
        return classify_rule(text), 0.5


__all__ = [
    "EventType",
    "EVENT_TYPES",
    "classify_rule",
    "classify_finbert",
    "classify_claude",
]
