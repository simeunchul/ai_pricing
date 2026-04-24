"""ETF iNAV deviation strategy (개인계정 단방향).

Logic:
  - Compute iNAV from basket of constituents.
  - Deviation = (ETF price - iNAV) / iNAV.
  - If deviation < -enter_threshold: BUY (ETF cheap vs basket; expect convergence).
  - If deviation > +enter_threshold and holding: SELL.
  - Flat at exit_threshold.

NOTE: In real LP operations, one would arbitrage both legs. For a retail
portfolio demo we only trade ETF and log the "basket shadow" decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class EtfInavArbitrage:
    enter_threshold: float = 0.003   # 30 bps
    exit_threshold: float = 0.0005   # 5 bps
    max_position: int = 10
    position: int = 0

    def decide(self, deviation: float) -> Signal:
        if self.position == 0:
            if deviation < -self.enter_threshold:
                return Signal.BUY
            if deviation > self.enter_threshold:
                return Signal.SELL  # short-sell if allowed; for cash account: just flat
        elif self.position > 0:
            if deviation > -self.exit_threshold:
                return Signal.SELL
        else:  # short
            if deviation < self.exit_threshold:
                return Signal.BUY
        return Signal.HOLD

    def apply(self, signal: Signal, qty: int = 1) -> None:
        if signal == Signal.BUY:
            self.position = min(self.position + qty, self.max_position)
        elif signal == Signal.SELL:
            self.position = max(self.position - qty, -self.max_position)


__all__ = ["Signal", "EtfInavArbitrage"]
