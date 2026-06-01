"""DualPaperState 저장/로드 단위 테스트 + is_market_open 헬퍼."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from autotrader.paper.dual_state import DualPaperState, Position


def _is_market_open(now):
    """Standalone copy of scripts/run_dual_paper_trading.is_market_open
    (script 모듈 import 회피용)."""
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    return (h, m) >= (9, 0) and (h, m) <= (15, 30)


def test_market_open_during_session():
    # 2026-04-30 (목) 10:00 — 장중
    assert _is_market_open(datetime(2026, 4, 30, 10, 0)) is True


def test_market_closed_before_open():
    # 2026-04-30 (목) 08:30 — 장 시작 전
    assert _is_market_open(datetime(2026, 4, 30, 8, 30)) is False


def test_market_closed_after_close():
    # 2026-04-30 (목) 16:00 — 장 마감 후
    assert _is_market_open(datetime(2026, 4, 30, 16, 0)) is False


def test_market_closed_weekend():
    # 2026-05-02 (토) 12:00
    assert _is_market_open(datetime(2026, 5, 2, 12, 0)) is False
    # 2026-05-03 (일) 12:00
    assert _is_market_open(datetime(2026, 5, 3, 12, 0)) is False


def test_market_edge_open_exact():
    # 09:00:00 정각 — open
    assert _is_market_open(datetime(2026, 4, 30, 9, 0)) is True


def test_market_edge_close_exact():
    # 15:30:00 정각 — still open (boundary inclusive)
    assert _is_market_open(datetime(2026, 4, 30, 15, 30)) is True
    # 15:31:00 — closed
    assert _is_market_open(datetime(2026, 4, 30, 15, 31)) is False


def test_fresh_state_defaults():
    s = DualPaperState(initial_cash=5_000_000)
    assert s.cash == 5_000_000
    assert s.positions == {}
    assert s.portfolio_peak == 5_000_000
    assert s.cooldown_remaining == 0
    assert s.run_count == 0


def test_save_and_load_roundtrip(tmp_path):
    s = DualPaperState(
        initial_cash=10_000_000, cash=8_500_000,
        portfolio_peak=10_500_000,
        cooldown_remaining=5, run_count=12,
        last_run="2026-04-30T09:05:00",
    )
    s.positions["005930"] = Position(qty=100, avg_entry=75000.0, entry_date="2026-04-30")
    s.positions["000660"] = Position(qty=50, avg_entry=280000.0, entry_date="2026-04-29")

    path = tmp_path / "state.json"
    s.save(path)
    assert path.exists()

    s2 = DualPaperState.load(path)
    assert s2.cash == 8_500_000
    assert s2.portfolio_peak == 10_500_000
    assert s2.cooldown_remaining == 5
    assert s2.run_count == 12
    assert len(s2.positions) == 2
    assert s2.positions["005930"].qty == 100
    assert s2.positions["005930"].avg_entry == 75000.0
    assert s2.positions["000660"].qty == 50


def test_load_missing_file_returns_fresh(tmp_path):
    path = tmp_path / "nonexistent.json"
    s = DualPaperState.load(path, initial_cash=20_000_000)
    assert s.cash == 20_000_000
    assert s.positions == {}
    assert s.run_count == 0


def test_load_corrupt_file_returns_fresh(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("not json content {[")
    s = DualPaperState.load(path, initial_cash=20_000_000)
    assert s.cash == 20_000_000
    assert s.positions == {}


def test_save_then_load_preserves_korean_names(tmp_path):
    s = DualPaperState()
    s.positions["005930"] = Position(qty=10, avg_entry=75000.0, entry_date="2026-04-30")
    path = tmp_path / "state.json"
    s.save(path)
    text = path.read_text(encoding='utf-8')
    # 정확한 데이터 보존
    assert "005930" in text
    s2 = DualPaperState.load(path)
    assert s2.positions["005930"].qty == 10
