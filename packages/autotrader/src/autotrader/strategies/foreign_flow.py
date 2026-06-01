"""Foreign-investor flow-following strategy.

Hypothesis: 외국인이 어제 많이 산 종목은 오늘도 오를 가능성이 높다.

Signal:
  flow_ratio = foreign_net_shares / volume   (그날 거래량 중 외국인 순매수 비중)
  flow_ratio > +enter_threshold  →  T+1 시가 매수
  flow_ratio < -enter_threshold  →  보유 시 즉시 청산
  보유일수 >= exit_after_days     →  자동 청산

Long-only (KIS 모의투자 공매도 불가). 동일 dataclass + duck-typed 인터페이스
규약을 EtfInavArbitrage 와 공유. 멀티 심볼 독립 포지션 관리.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autotrader.strategies.etf_inav_arb import Signal


@dataclass
class ForeignFlowFollow:
    enter_threshold: float = 0.05            # 거래대금의 5%
    exit_after_days: int = 2                 # 보유 만기 (일)
    qty_per_signal: int = 10                 # 한 번 신호에 매수할 수량
    max_position_per_symbol: int = 50        # 종목별 포지션 캡
    positions: dict[str, int] = field(default_factory=dict)
    days_held: dict[str, int] = field(default_factory=dict)

    def decide(self, symbol: str, flow_ratio: float) -> tuple[Signal, int]:
        """단일 심볼 시그널.

        우선순위:
          1) 보유 중 + 만기 도달 → 전량 청산
          2) 보유 중 + 외국인 매도 강세 → 전량 청산 (조기 손절)
          3) 미보유 또는 캡 미도달 + 외국인 매수 강세 → 매수
          4) 그 외 → HOLD
        """
        pos = self.positions.get(symbol, 0)
        held = self.days_held.get(symbol, 0)

        if pos > 0 and held >= self.exit_after_days:
            return Signal.SELL, pos

        if pos > 0 and flow_ratio < -self.enter_threshold:
            return Signal.SELL, pos

        if flow_ratio > self.enter_threshold and pos < self.max_position_per_symbol:
            qty = min(self.qty_per_signal, self.max_position_per_symbol - pos)
            if qty > 0:
                return Signal.BUY, qty

        return Signal.HOLD, 0

    def apply(self, symbol: str, signal: Signal, qty: int) -> None:
        """포지션 갱신. days_held 는 advance_day() 가 별도로 누적."""
        if signal == Signal.HOLD or qty <= 0:
            return
        pos = self.positions.get(symbol, 0)
        if signal == Signal.BUY:
            new_pos = min(pos + qty, self.max_position_per_symbol)
            self.positions[symbol] = new_pos
            if pos == 0:
                # 첫 진입 시 보유일수 0 으로 초기화
                self.days_held[symbol] = 0
        elif signal == Signal.SELL:
            new_pos = max(pos - qty, 0)
            self.positions[symbol] = new_pos
            if new_pos == 0:
                self.days_held.pop(symbol, None)

    def advance_day(self) -> None:
        """하루 경과. 모든 보유 포지션의 days_held +1."""
        for sym in list(self.days_held.keys()):
            if self.positions.get(sym, 0) > 0:
                self.days_held[sym] = self.days_held.get(sym, 0) + 1
            else:
                self.days_held.pop(sym, None)


__all__ = ["ForeignFlowFollow"]
