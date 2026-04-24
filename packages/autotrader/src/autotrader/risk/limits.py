"""Risk guards. Required by runner before every order submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time


@dataclass
class RiskLimits:
    max_position_pct: float = 0.20       # max position value / equity
    daily_loss_pct: float = -0.015       # stop trading below -1.5%
    max_consecutive_errors: int = 3
    no_trade_start: time = time(9, 0)
    no_trade_end_open: time = time(9, 10)   # skip first 10min
    no_trade_start_close: time = time(15, 20)  # skip last 10min
    no_trade_end: time = time(15, 30)


@dataclass
class RiskState:
    equity_open: float = 0.0
    equity_now: float = 0.0
    position_value: float = 0.0
    error_count: int = 0


def check(state: RiskState, limits: RiskLimits, now: datetime | None = None) -> tuple[bool, str]:
    """Return (ok, reason). ok=False means trading must halt."""
    now = now or datetime.now()
    t = now.time()

    # Time-of-day guards
    if t < limits.no_trade_start or t > limits.no_trade_end:
        return False, f"outside trading hours: {t}"
    if limits.no_trade_start <= t < limits.no_trade_end_open:
        return False, "opening window (first 10 min) blocked"
    if limits.no_trade_start_close <= t < limits.no_trade_end:
        return False, "closing window (last 10 min) blocked"

    # Daily loss
    if state.equity_open > 0:
        day_ret = (state.equity_now - state.equity_open) / state.equity_open
        if day_ret < limits.daily_loss_pct:
            return False, f"daily loss hit: {day_ret:.2%}"

    # Position size
    if state.equity_now > 0:
        pos_pct = state.position_value / state.equity_now
        if pos_pct > limits.max_position_pct:
            return False, f"position exceeds cap: {pos_pct:.2%}"

    # Consecutive errors
    if state.error_count >= limits.max_consecutive_errors:
        return False, f"error count {state.error_count}"

    return True, "ok"


__all__ = ["RiskLimits", "RiskState", "check"]
