"""Foreign-flow proportional sizing strategy.

ForeignFlowFollow 의 변형:
  - 고정 qty 가 아니라 flow_ratio 절대값에 비례해서 매수/매도 수량 결정.
  - 매도 시 전량 청산이 아닌 부분 청산 (외국인 매도 강도에 비례).
  - 만기 청산 룰 없음 (외국인 매도 신호 또는 캡 도달까지 보유).

Sizing rule:
  qty = round(|flow_ratio| * sizing_factor)

예시 (sizing_factor=100):
  flow_ratio = +0.10 → 10주 매수
  flow_ratio = +0.02 → 2주 매수
  flow_ratio = -0.10 → 보유분 중 10주 매도 (보유 < 10이면 전량)
  flow_ratio = -0.02 → 보유분 중 2주 매도

Long-only (KIS 모의투자 공매도 불가). 매도가 보유량 초과 시 전량 청산
(잉여 신호는 무시; 신규 short 진입 안 함).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autotrader.strategies.etf_inav_arb import Signal


@dataclass
class ForeignFlowProportional:
    sizing_factor: float = 200.0           # qty = round(|flow_ratio| * sizing_factor)
    min_threshold: float = 0.01            # 노이즈 컷 — 미만이면 신호 무시
    max_position_per_symbol: int = 1_000   # 사실상 비활성 (sizing이 자율)
    positions: dict[str, int] = field(default_factory=dict)
    days_held: dict[str, int] = field(default_factory=dict)

    def _qty_for(self, flow_ratio: float) -> int:
        if abs(flow_ratio) < self.min_threshold:
            return 0
        return max(1, round(abs(flow_ratio) * self.sizing_factor))

    def decide(self, symbol: str, flow_ratio: float) -> tuple[Signal, int]:
        pos = self.positions.get(symbol, 0)

        # 매도 신호 (외국인 순매도)
        if flow_ratio < -self.min_threshold and pos > 0:
            target = self._qty_for(flow_ratio)
            actual = min(target, pos)   # 보유분 초과 매도 신호 → 전량 청산
            if actual > 0:
                return Signal.SELL, actual
            return Signal.HOLD, 0

        # 매수 신호 (외국인 순매수)
        if flow_ratio > self.min_threshold and pos < self.max_position_per_symbol:
            target = self._qty_for(flow_ratio)
            headroom = self.max_position_per_symbol - pos
            actual = min(target, headroom)
            if actual > 0:
                return Signal.BUY, actual
            return Signal.HOLD, 0

        return Signal.HOLD, 0

    def apply(self, symbol: str, signal: Signal, qty: int) -> None:
        if signal == Signal.HOLD or qty <= 0:
            return
        pos = self.positions.get(symbol, 0)
        if signal == Signal.BUY:
            new_pos = min(pos + qty, self.max_position_per_symbol)
            self.positions[symbol] = new_pos
            if pos == 0:
                self.days_held[symbol] = 0
        elif signal == Signal.SELL:
            new_pos = max(pos - qty, 0)
            self.positions[symbol] = new_pos
            if new_pos == 0:
                self.days_held.pop(symbol, None)

    def advance_day(self) -> None:
        """인터페이스 호환용 (만기 청산 없음 — days_held 카운터만 유지)."""
        for sym in list(self.days_held.keys()):
            if self.positions.get(sym, 0) > 0:
                self.days_held[sym] = self.days_held.get(sym, 0) + 1
            else:
                self.days_held.pop(sym, None)


__all__ = ["ForeignFlowProportional"]
