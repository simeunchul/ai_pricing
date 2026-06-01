"""Dual Confirmation paper trading 상태 저장/로드.

매일 1회 실행되는 strategy 의 cross-day state:
  - positions: 보유 종목별 qty, avg_entry, entry_date
  - portfolio_peak: 자본 사상 최고점 (MDD cap 기준)
  - cooldown_remaining: MDD cap 발동 후 남은 cooldown 일수
  - last_run: 마지막 실행 시각
  - cash: 보유 cash (paper 모드에서 추적용)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Position:
    qty: int
    avg_entry: float
    entry_date: str                  # ISO date
    kospi_at_entry: float | None = None   # 매수 시점 KOSPI 지수 (underperform 룰 #5)


@dataclass
class DualPaperState:
    initial_cash: float = 10_000_000.0
    cash: float | None = None              # None → __post_init__ 에서 initial_cash 로 채움
    positions: dict[str, Position] = field(default_factory=dict)
    portfolio_peak: float | None = None    # None → initial_cash 로 채움
    cooldown_remaining: int = 0
    last_run: str | None = None
    run_count: int = 0

    def __post_init__(self):
        if self.cash is None:
            self.cash = self.initial_cash
        if self.portfolio_peak is None:
            self.portfolio_peak = self.initial_cash

    def to_dict(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "portfolio_peak": self.portfolio_peak,
            "cooldown_remaining": self.cooldown_remaining,
            "last_run": self.last_run,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DualPaperState":
        positions = {
            sym: Position(**v) for sym, v in d.get("positions", {}).items()
        }
        return cls(
            initial_cash=d.get("initial_cash", 10_000_000.0),
            cash=d.get("cash", d.get("initial_cash", 10_000_000.0)),
            positions=positions,
            portfolio_peak=d.get("portfolio_peak", d.get("initial_cash", 10_000_000.0)),
            cooldown_remaining=d.get("cooldown_remaining", 0),
            last_run=d.get("last_run"),
            run_count=d.get("run_count", 0),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        logger.info(f"state saved → {path}")

    @classmethod
    def load(cls, path: Path, initial_cash: float = 10_000_000.0) -> "DualPaperState":
        if not path.exists():
            logger.info(f"state file not found at {path} — fresh start with {initial_cash:,.0f}원")
            return cls(initial_cash=initial_cash, cash=initial_cash, portfolio_peak=initial_cash)
        try:
            d = json.loads(path.read_text())
            return cls.from_dict(d)
        except Exception as e:
            logger.error(f"state load failed: {e} — fresh start")
            return cls(initial_cash=initial_cash, cash=initial_cash, portfolio_peak=initial_cash)


__all__ = ["Position", "DualPaperState"]
