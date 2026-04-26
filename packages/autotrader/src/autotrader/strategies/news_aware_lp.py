"""B3 → ETF LP integration (#4 통합).

기본 Avellaneda-Stoikov LP 위에 B3 의 macro shock 신호로 σ 동적 조정.
σ 가 spread 결정 → spread = γ·σ²·(T-t) + (2/γ)·log(1+γ/k).
σ ↑ → spread ↑ → 호가 멀리 → 위기 시 보수적 LP.

실무 의미:
  평소: σ_baseline → 좁은 spread → 많은 거래
  Macro shock 감지: σ_baseline × shock_multiplier → 넓은 spread → 위험 회피
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# B3 import
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from ai_pricing.news_iv.classify import classify_rule, EventType
from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES, dominant_event

from autotrader.strategies.avellaneda_stoikov import (
    ASConfig, ASState, Quote, compute_quote,
)


@dataclass
class NewsAwareASConfig(ASConfig):
    """ASConfig + news shock multiplier knobs."""
    sigma_baseline: float = 0.01          # 평소 σ (보통 1분봉 0.01)
    shock_multiplier: float = 3.0         # macro shock 시 σ × 3
    enter_threshold_multiplier: float = 0.5  # shock 시 enter_threshold × 0.5 (더 민감)


HIGH_VOL_EVENTS: set[str] = {"earnings_miss", "macro_shock", "regulatory", "mna"}


class NewsAwareASStrategy:
    """Avellaneda-Stoikov + B3 news 신호 wrapper.

    Hybrid pattern:
      - 매 quote tick: news 분류 → if shock: σ 부스트 → AS 호가 재계산
      - 평소: ASConfig.sigma_baseline 유지
    """

    def __init__(self, cfg: NewsAwareASConfig | None = None):
        self.cfg = cfg or NewsAwareASConfig()
        self._last_event: EventType = "neutral"
        self._last_sigma: float = self.cfg.sigma_baseline
        self._inner_cfg: ASConfig = self._build_inner_cfg(self._last_sigma)
        self.state = ASState()

    def _build_inner_cfg(self, sigma: float) -> ASConfig:
        return ASConfig(
            gamma=self.cfg.gamma,
            sigma=sigma,
            k=self.cfg.k,
            inventory_limit=self.cfg.inventory_limit,
            quote_size=self.cfg.quote_size,
            tick_size=self.cfg.tick_size,
        )

    def update_news(self, news_texts: list[str]) -> tuple[EventType, float]:
        """Reclassify recent news → adjust σ for AS strategy."""
        events = [(classify_rule(t), 1.0) for t in news_texts]
        ev, conf = dominant_event(events)
        self._last_event = ev

        if ev in HIGH_VOL_EVENTS:
            new_sigma = self.cfg.sigma_baseline * self.cfg.shock_multiplier
        else:
            new_sigma = self.cfg.sigma_baseline

        if abs(new_sigma - self._last_sigma) > 1e-9:
            self._last_sigma = new_sigma
            self._inner_cfg = self._build_inner_cfg(new_sigma)
        return ev, conf

    def quote(self, mid: float, t_norm: float, inventory: int) -> Quote:
        """Get quote at time t (∈ [0, 1]) given current inventory."""
        self.state.inventory = inventory
        t_remaining = max(0.0, 1.0 - t_norm)
        return compute_quote(self.state, mid, t_remaining, self._inner_cfg)

    @property
    def last_event(self) -> EventType:
        return self._last_event

    @property
    def last_sigma(self) -> float:
        return self._last_sigma

    @property
    def shock_active(self) -> bool:
        return self._last_event in HIGH_VOL_EVENTS


__all__ = ["NewsAwareASConfig", "NewsAwareASStrategy", "HIGH_VOL_EVENTS"]
