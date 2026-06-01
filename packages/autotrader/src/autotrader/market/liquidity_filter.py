"""유동성 필터 — 거래대금 + 호가 스프레드 기반 종목 진입 적격 판정.

Phase 1 (거래대금): KIS foreign-institution-total 응답의 acml_vol × price 로 계산
Phase 2 (호가 스프레드): KIS asking-price endpoint 호출 후 (ask - bid) / mid 계산

진입 적격 조건:
  거래대금 ≥ MIN_TRADING_VALUE_KRW (default 5억원)
  호가 스프레드 ≤ MAX_SPREAD_BPS (default 30bps = 0.3%)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# 우리 운용 기준 (자본 1000만원, slot 자본 ~333만원 가정):
#   매수 자본 333만 / 거래대금 5억 = 0.67% — 시장 충격 미미
MIN_TRADING_VALUE_KRW = 500_000_000      # 5억원
MAX_SPREAD_BPS = 30                       # 0.3%


@dataclass
class LiquidityCheck:
    symbol: str
    passed: bool
    trading_value: float          # 당일 누적 거래대금 (원)
    spread_bps: float | None      # 호가 스프레드 (bps); None = 미조회
    reason: str                   # 통과/탈락 이유


def check_trading_value(
    symbol: str, price: float, acml_vol: float,
    min_krw: float = MIN_TRADING_VALUE_KRW,
) -> tuple[bool, float, str]:
    """Phase 1: 거래대금 필터.

    Returns: (passed, trading_value, reason)
    """
    trading_value = price * acml_vol
    if trading_value < min_krw:
        return False, trading_value, (
            f"거래대금 부족: {trading_value/1e8:.2f}억 < {min_krw/1e8:.0f}억 임계"
        )
    return True, trading_value, f"거래대금 OK: {trading_value/1e8:.2f}억"


def parse_asking_price(payload: dict) -> dict:
    """KIS asking-price 응답 → bid/ask/mid/spread_bps.

    Returns dict with keys: ask, bid, mid, spread, spread_bps,
                           total_ask_qty, total_bid_qty
    None 값이면 파싱 실패.
    """
    out = payload.get("output1", {})
    if not isinstance(out, dict) or not out:
        return {}

    def _to_int(s) -> int:
        if s is None:
            return 0
        if isinstance(s, (int, float)):
            return int(s)
        s = str(s).strip().replace(",", "")
        if not s or s == "-":
            return 0
        try:
            return int(float(s))
        except ValueError:
            return 0

    ask = _to_int(out.get("askp1"))
    bid = _to_int(out.get("bidp1"))
    if ask <= 0 or bid <= 0:
        return {}

    mid = (ask + bid) / 2.0
    spread = ask - bid
    spread_bps = (spread / mid * 10_000) if mid > 0 else float("inf")

    return {
        "ask": ask,
        "bid": bid,
        "mid": mid,
        "spread": spread,
        "spread_bps": spread_bps,
        "total_ask_qty": _to_int(out.get("total_askp_rsqn")),
        "total_bid_qty": _to_int(out.get("total_bidp_rsqn")),
    }


def check_spread(
    symbol: str, asking_payload: dict,
    max_bps: float = MAX_SPREAD_BPS,
) -> tuple[bool, float | None, str]:
    """Phase 2: 호가 스프레드 필터.

    Returns: (passed, spread_bps, reason)
    """
    parsed = parse_asking_price(asking_payload)
    if not parsed:
        return False, None, "호가 응답 파싱 실패"

    bps = parsed["spread_bps"]
    if bps > max_bps:
        return False, bps, (
            f"스프레드 과대: {bps:.1f}bps > {max_bps:.0f}bps 한도"
        )
    return True, bps, f"스프레드 OK: {bps:.1f}bps"


def evaluate_liquidity(
    symbol: str, price: float, acml_vol: float,
    asking_payload: dict | None = None,
    min_trading_value_krw: float = MIN_TRADING_VALUE_KRW,
    max_spread_bps: float = MAX_SPREAD_BPS,
) -> LiquidityCheck:
    """거래대금 + (선택)호가 스프레드 통합 평가.

    asking_payload=None 이면 Phase 1 만 (거래대금 단독). 이 모드에서는
    spread_bps=None.
    """
    # Phase 1
    ok_v, tv, reason_v = check_trading_value(
        symbol, price, acml_vol, min_krw=min_trading_value_krw
    )
    if not ok_v:
        return LiquidityCheck(
            symbol=symbol, passed=False,
            trading_value=tv, spread_bps=None, reason=reason_v,
        )

    # Phase 2 (선택)
    if asking_payload is None:
        return LiquidityCheck(
            symbol=symbol, passed=True,
            trading_value=tv, spread_bps=None, reason=reason_v,
        )

    ok_s, bps, reason_s = check_spread(
        symbol, asking_payload, max_bps=max_spread_bps
    )
    if not ok_s:
        return LiquidityCheck(
            symbol=symbol, passed=False,
            trading_value=tv, spread_bps=bps,
            reason=f"{reason_v}; {reason_s}",
        )

    return LiquidityCheck(
        symbol=symbol, passed=True,
        trading_value=tv, spread_bps=bps,
        reason=f"{reason_v}; {reason_s}",
    )


__all__ = [
    "LiquidityCheck",
    "check_trading_value",
    "check_spread",
    "parse_asking_price",
    "evaluate_liquidity",
    "MIN_TRADING_VALUE_KRW",
    "MAX_SPREAD_BPS",
]
