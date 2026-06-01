"""Foreign + Institution dual-confirmation strategy.

가설 (실증 연구 기반): 외국인과 기관이 동시에 같은 방향으로 매매할 때 신호가
가장 강하다. 한쪽만 매수하면 noise 가 크다 (예: 외국인 매수 vs 기관 매도 →
서로 반대 의견 → 단기 방향성 불명확).

Signal:
  flow_ratio  = 외국인 순매수 / 거래량
  inst_ratio  = 기관 순매수 / 거래량

  매수: flow_ratio > +threshold AND inst_ratio > +threshold
  매도: flow_ratio < -threshold AND inst_ratio < -threshold (보유 중)
  반대 신호 (예: foreign +, inst -) 또는 한쪽만 강함 → HOLD

  매매 강도: min(|flow_ratio|, |inst_ratio|) × sizing_factor
            = 더 약한 쪽 기준 (보수적, conservative confirmation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autotrader.strategies.etf_inav_arb import Signal


@dataclass
class ForeignInstFlowFollow:
    enter_threshold: float = 0.05            # 외국인·기관 둘 다 이 임계 초과해야 진입
    sizing_factor: float = 200.0             # qty = round(min_ratio × sizing_factor)
    min_threshold: float = 0.01              # 한 쪽이라도 이 미만이면 노이즈 → HOLD
    max_position_per_symbol: int = 10_000
    positions: dict[str, int] = field(default_factory=dict)
    days_held: dict[str, int] = field(default_factory=dict)

    def _qty_for(self, flow_ratio: float, inst_ratio: float) -> int:
        weaker = min(abs(flow_ratio), abs(inst_ratio))
        if weaker < self.min_threshold:
            return 0
        return max(1, round(weaker * self.sizing_factor))

    def decide(self, symbol: str, flow_ratio: float, inst_ratio: float) -> tuple[Signal, int]:
        pos = self.positions.get(symbol, 0)

        # 양쪽 신호 강도가 모두 노이즈 컷 이상인지 확인
        if abs(flow_ratio) < self.min_threshold or abs(inst_ratio) < self.min_threshold:
            return Signal.HOLD, 0

        # 부호 일치 필요 (둘 다 + 또는 둘 다 -)
        if flow_ratio * inst_ratio < 0:
            return Signal.HOLD, 0

        # 매도 신호: 둘 다 -enter_threshold 이하 + 보유 중
        if (
            flow_ratio < -self.enter_threshold
            and inst_ratio < -self.enter_threshold
            and pos > 0
        ):
            target = self._qty_for(flow_ratio, inst_ratio)
            actual = min(target, pos)
            if actual > 0:
                return Signal.SELL, actual

        # 매수 신호: 둘 다 +enter_threshold 이상 + 캡 미도달
        if (
            flow_ratio > self.enter_threshold
            and inst_ratio > self.enter_threshold
            and pos < self.max_position_per_symbol
        ):
            target = self._qty_for(flow_ratio, inst_ratio)
            headroom = self.max_position_per_symbol - pos
            actual = min(target, headroom)
            if actual > 0:
                return Signal.BUY, actual

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
        for sym in list(self.days_held.keys()):
            if self.positions.get(sym, 0) > 0:
                self.days_held[sym] = self.days_held.get(sym, 0) + 1
            else:
                self.days_held.pop(sym, None)


__all__ = ["ForeignInstFlowFollow"]
