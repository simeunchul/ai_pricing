"""ETF iNAV deviation strategy (개인계정 단방향 + 인버스 ETF 합성 short).

Logic:
  - Compute iNAV from basket of constituents.
  - Deviation = (ETF price - iNAV) / iNAV.
  - If deviation < -enter_threshold: BUY ETF (cheap; expect convergence).
  - If deviation > +enter_threshold: BUY inverse ETF (expensive; synthetic short).
  - Exit when deviation crosses back through ±exit_threshold.

Aggressive sizing:
  qty = min(int(|dev| / enter_threshold) * qty_per_step, max_position)
  Examples (qty_per_step=5, enter_threshold=30bps):
      30 bps  →  5 shares
      60 bps  → 10 shares
      90 bps  → 15 shares
      300 bps → 50 shares (cap)

Open/exit positions are full-fill: enter with sized qty, exit closes everything.
Pyramiding (adding to a winning position when deviation widens further) is supported
inside max_position cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class EtfInavArbitrage:
    enter_threshold: float = 0.003   # 30 bps
    exit_threshold: float = 0.0005   # 5 bps
    qty_per_step: int = 5            # aggressive sizing multiplier per threshold step
    max_position: int = 50           # cap per leg (ETF or inverse)
    position: int = 0                # ETF long position (shares)
    inverse_position: int = 0        # inverse ETF long (= synthetic short on underlying)

    def _qty_for(self, deviation: float) -> int:
        # 1e-9 epsilon: 0.009 / 0.003 lands at 2.9999… in IEEE-754; without it
        # int() truncates a clean 3-step deviation down to 2 steps.
        steps = int(abs(deviation) / self.enter_threshold + 1e-9)
        if steps <= 0:
            return 0
        return min(steps * self.qty_per_step, self.max_position)

    def decide(self, deviation: float) -> Signal:
        """Backward-compatible single-share decision (no inverse routing)."""
        if self.position == 0:
            if deviation < -self.enter_threshold:
                return Signal.BUY
            if deviation > self.enter_threshold:
                return Signal.SELL
        elif self.position > 0:
            if deviation > -self.exit_threshold:
                return Signal.SELL
        else:
            if deviation < self.exit_threshold:
                return Signal.BUY
        return Signal.HOLD

    def decide_aggressive(self, deviation: float) -> tuple[Signal, int, bool]:
        """Returns (signal, qty, use_inverse).

        use_inverse=True  → trade the inverse ETF (BUY = open synthetic short,
                            SELL = close synthetic short).
        use_inverse=False → trade the ETF directly (BUY = long, SELL = exit long).
        """
        target_qty = self._qty_for(deviation)

        if self.position > 0:
            if deviation > -self.exit_threshold:
                return Signal.SELL, self.position, False
            if deviation < -self.enter_threshold:
                extra = min(
                    max(0, target_qty - self.position),
                    self.max_position - self.position,
                )
                if extra > 0:
                    return Signal.BUY, extra, False
            return Signal.HOLD, 0, False

        if self.inverse_position > 0:
            if deviation < self.exit_threshold:
                return Signal.SELL, self.inverse_position, True
            if deviation > self.enter_threshold:
                extra = min(
                    max(0, target_qty - self.inverse_position),
                    self.max_position - self.inverse_position,
                )
                if extra > 0:
                    return Signal.BUY, extra, True
            return Signal.HOLD, 0, False

        if deviation < -self.enter_threshold:
            return Signal.BUY, target_qty, False
        if deviation > self.enter_threshold:
            return Signal.BUY, target_qty, True
        return Signal.HOLD, 0, False

    def decide_with_fair(self, dev_basket: float, fair_dev: float,
                          fair_blend: float = 0.3) -> tuple[Signal, int, bool]:
        """Phase A4 — basket dev + 모델 fair dev 조합 결정.

        combined_dev = dev_basket * (1 - fair_blend) + fair_dev * fair_blend

        fair_blend=0   → 기존 decide_aggressive 와 동일 (basket 만)
        fair_blend=1   → 모델만 사용 (basket 무시)
        fair_blend=0.3 → 보수적 ensemble (검증 부족 모델에 30% 가중)

        Returns: (signal, qty, use_inverse) — decide_aggressive 와 동일 형식.
        """
        combined = dev_basket * (1 - fair_blend) + fair_dev * fair_blend
        return self.decide_aggressive(combined)

    def apply(self, signal: Signal, qty: int = 1) -> None:
        if signal == Signal.BUY:
            self.position = min(self.position + qty, self.max_position)
        elif signal == Signal.SELL:
            self.position = max(self.position - qty, -self.max_position)

    def apply_aggressive(self, signal: Signal, qty: int, use_inverse: bool) -> None:
        if signal == Signal.HOLD or qty <= 0:
            return
        if use_inverse:
            if signal == Signal.BUY:
                self.inverse_position = min(self.inverse_position + qty, self.max_position)
            elif signal == Signal.SELL:
                self.inverse_position = max(self.inverse_position - qty, 0)
        else:
            if signal == Signal.BUY:
                self.position = min(self.position + qty, self.max_position)
            elif signal == Signal.SELL:
                self.position = max(self.position - qty, 0)


__all__ = ["Signal", "EtfInavArbitrage"]
