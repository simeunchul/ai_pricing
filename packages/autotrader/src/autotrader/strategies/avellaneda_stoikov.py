"""Avellaneda-Stoikov 양방향 호가 + 재고 페널티 LP 전략.

원논문: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
       Quantitative Finance 8(3), 217-224.

핵심 수식:
  reservation_price = mid - q * γ * σ² * (T - t)
  optimal_spread    = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
  bid = reservation - spread/2
  ask = reservation + spread/2

  q       : current inventory (signed)
  γ (gamma): risk aversion (높을수록 재고 회피)
  σ       : asset volatility (1-step return)
  T - t   : remaining time (normalized to [0, 1])
  k       : intensity of order arrival (Poisson rate scale)

Why this beats iNAV-only single-direction strategy:
  - LP 는 양방향 호가 제공 → bid-ask spread 가 곧 수익
  - 재고 (q) 가 한 방향으로 쌓이면 reservation price 가 자동 조정 → 자연스러운 mean-revert
  - σ²(T-t) 가 risk 를 시간 따라 줄임 → 만기 임박 시 spread 좁아짐

본 모듈은 백테스트 + 시뮬용. 실 KIS 호가 제출 코드는 별도.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ASConfig:
    """Avellaneda-Stoikov 파라미터.

    All times normalized to [0, 1] for one trading session.
    """
    gamma: float = 0.1               # 위험회피 계수
    sigma: float = 0.01              # 1-step return std (e.g. 1분봉 기준 0.01 = 1%)
    k: float = 1.5                   # 주문 도착 강도 scale
    inventory_limit: int = 50        # 절대 inventory 상한
    quote_size: int = 1              # 호가 1건 당 수량
    tick_size: float = 0.001         # 최소 호가 단위 (proportional to mid)


@dataclass
class Quote:
    bid: float
    ask: float
    reservation: float
    spread: float
    inventory: int
    t_remaining: float


@dataclass
class ASState:
    """LP 의 현재 상태."""
    inventory: int = 0
    cash: float = 0.0
    realized_pnl: float = 0.0
    n_buys: int = 0
    n_sells: int = 0


def compute_quote(state: ASState, mid: float, t_remaining: float,
                   cfg: ASConfig) -> Quote:
    """현재 mid 와 t_remaining 으로 양방향 optimal bid/ask 계산.

    t_remaining ∈ [0, 1] (1=session 시작, 0=session 종료).
    """
    q = state.inventory
    sigma2_T = cfg.sigma ** 2 * t_remaining

    # reservation price (재고 페널티)
    reservation = mid - q * cfg.gamma * sigma2_T

    # optimal spread (Avellaneda-Stoikov closed-form)
    inner = 1.0 + cfg.gamma / cfg.k
    spread = cfg.gamma * sigma2_T + (2.0 / cfg.gamma) * math.log(inner)

    bid = reservation - spread / 2.0
    ask = reservation + spread / 2.0

    # tick rounding (상대적)
    tick = max(cfg.tick_size * mid, 1e-9)
    bid = math.floor(bid / tick) * tick
    ask = math.ceil(ask / tick) * tick

    return Quote(
        bid=round(bid, 6),
        ask=round(ask, 6),
        reservation=round(reservation, 6),
        spread=round(spread, 6),
        inventory=q,
        t_remaining=round(t_remaining, 4),
    )


def fill_check(quote: Quote, mid_next: float, cfg: ASConfig,
                rng_uniform: float | None = None) -> Literal["bid", "ask", "none"]:
    """단순 fill model — 다음 mid 가 우리 bid 까지 내려가면 BUY 체결,
    ask 까지 올라가면 SELL 체결. 한 번에 하나만 체결로 가정.

    더 현실적: Poisson arrival with intensity decreasing in spread distance.
    """
    if mid_next <= quote.bid:
        return "bid"
    if mid_next >= quote.ask:
        return "ask"
    return "none"


def update_state(state: ASState, fill: str, quote: Quote, qty: int = 1) -> None:
    """체결에 따라 LP state 업데이트."""
    if fill == "bid":
        state.inventory += qty
        state.cash -= quote.bid * qty
        state.n_buys += 1
    elif fill == "ask":
        state.inventory -= qty
        state.cash += quote.ask * qty
        state.n_sells += 1


def mark_to_market(state: ASState, mid: float) -> float:
    """현재 mid 에서 LP 의 mark-to-market value (cash + inventory × mid)."""
    return state.cash + state.inventory * mid


__all__ = [
    "ASConfig", "ASState", "Quote",
    "compute_quote", "fill_check", "update_state", "mark_to_market",
]
