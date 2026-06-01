from autotrader.market.inav import InavEstimator, deviation
from autotrader.strategies.etf_inav_arb import EtfInavArbitrage, Signal
from autotrader.strategies.avellaneda_stoikov import (
    ASConfig, ASState, Quote, compute_quote, fill_check, update_state, mark_to_market,
)
from autotrader.risk.limits import RiskLimits, RiskState, check
from datetime import datetime


def test_inav_basic():
    est = InavEstimator(constituents={"A": 0.5, "B": 0.5})
    assert est.estimate({"A": 100, "B": 200}) == 150.0


def test_deviation_sign():
    assert deviation(101, 100) > 0
    assert deviation(99, 100) < 0
    assert deviation(100, 100) == 0


def test_inav_calibration():
    est = InavEstimator(constituents={"A": 0.5, "B": 0.5})
    assert not est.calibrated
    factor = est.calibrate(etf_price=300, prices={"A": 100, "B": 200})
    assert factor == 2.0
    assert est.calibrated
    assert est.estimate({"A": 100, "B": 200}) == 300.0
    assert abs(est.estimate({"A": 101, "B": 202}) - 303.0) < 1e-6


def test_inav_partial_coverage():
    est = InavEstimator(constituents={"A": 0.3, "B": 0.2, "C": 0.1}, cash_weight=0.4)
    raw = est._raw_basket({"A": 100, "B": 200, "C": 300})
    assert abs(raw - (100.0 / 0.6)) < 1e-6
    est.calibrate(etf_price=200, prices={"A": 100, "B": 200, "C": 300})
    assert abs(est.estimate({"A": 100, "B": 200, "C": 300}) - 200.0) < 1e-6


def test_inav_calibrated_dev_zero_at_t0():
    est = InavEstimator(constituents={"A": 1.0})
    est.calibrate(etf_price=100, prices={"A": 50})
    assert deviation(100, est.estimate({"A": 50})) == 0.0


def test_inav_calibration_deferred_when_any_leg_missing():
    """100% coverage 강제 — 1개라도 빠지면 calibrate() 실패 (factor=0)."""
    est = InavEstimator(constituents={"A": 0.5, "B": 0.3, "C": 0.2})
    factor = est.calibrate(etf_price=200, prices={"A": 100, "B": 200})
    assert factor == 0.0
    assert not est.calibrated
    # C 가 들어와야 비로소 calibration 성공
    factor = est.calibrate(etf_price=200, prices={"A": 100, "B": 200, "C": 50})
    assert factor > 0
    assert est.calibrated


def test_inav_estimate_returns_zero_if_calibrated_leg_missing():
    """calibration 후 leg 누락 시 estimate=0 (= 매매 신호 없음)."""
    est = InavEstimator(constituents={"A": 0.5, "B": 0.5})
    est.calibrate(etf_price=300, prices={"A": 100, "B": 200})
    # 정상 — 둘 다 있음
    assert est.estimate({"A": 100, "B": 200}) == 300.0
    # B 누락 → 0
    assert est.estimate({"A": 100}) == 0.0
    # B 가격 0 → 0 (zero/None 모두 missing 취급)
    assert est.estimate({"A": 100, "B": 0}) == 0.0
    # 추가 leg 가 들어와도 frozen leg-set 만 사용 — 정상 동작
    assert est.estimate({"A": 100, "B": 200, "C": 999}) == 300.0


def test_inav_calibrated_legs_property():
    est = InavEstimator(constituents={"X": 0.4, "Y": 0.6})
    est.calibrate(etf_price=100, prices={"X": 50, "Y": 100})
    assert set(est.calibrated_legs) == {"X", "Y"}


def test_deviation_zero_when_inav_zero():
    """estimate() 가 0 반환 시 deviation() 도 0 → strategy HOLD."""
    assert deviation(101, 0) == 0.0
    assert deviation(101, -5) == 0.0


def test_strategy_flat_to_buy_when_cheap():
    s = EtfInavArbitrage(enter_threshold=0.003)
    sig = s.decide(-0.005)
    assert sig == Signal.BUY


def test_strategy_exits_on_convergence():
    s = EtfInavArbitrage(enter_threshold=0.003, exit_threshold=0.0005)
    s.apply(Signal.BUY)  # long 1
    sig = s.decide(0.0)
    assert sig == Signal.SELL


def test_qty_ladder_scales_with_deviation():
    s = EtfInavArbitrage(enter_threshold=0.003, qty_per_step=5, max_position=50)
    assert s._qty_for(0.0010) == 0      # below threshold
    assert s._qty_for(0.0030) == 5      # 1 step
    assert s._qty_for(0.0060) == 10     # 2 steps
    assert s._qty_for(0.0090) == 15     # 3 steps
    assert s._qty_for(0.0300) == 50     # 10 steps → cap
    assert s._qty_for(0.0500) == 50     # capped


def test_decide_aggressive_buys_etf_when_cheap():
    s = EtfInavArbitrage(qty_per_step=5)
    sig, qty, inv = s.decide_aggressive(-0.006)  # 60bps cheap
    assert sig == Signal.BUY
    assert qty == 10
    assert inv is False


def test_decide_aggressive_buys_inverse_when_expensive():
    s = EtfInavArbitrage(qty_per_step=5)
    sig, qty, inv = s.decide_aggressive(0.009)  # 90bps expensive
    assert sig == Signal.BUY
    assert qty == 15
    assert inv is True  # synthetic short via inverse ETF


def test_decide_aggressive_holds_inside_band():
    s = EtfInavArbitrage(qty_per_step=5)
    sig, qty, inv = s.decide_aggressive(0.001)  # 10bps, below 30bps enter
    assert sig == Signal.HOLD
    assert qty == 0


def test_decide_aggressive_exits_long_on_convergence():
    s = EtfInavArbitrage(qty_per_step=5)
    s.apply_aggressive(Signal.BUY, 10, False)  # long 10 ETF
    sig, qty, inv = s.decide_aggressive(0.0)
    assert sig == Signal.SELL
    assert qty == 10  # close all
    assert inv is False


def test_decide_aggressive_exits_inverse_on_convergence():
    s = EtfInavArbitrage(qty_per_step=5)
    s.apply_aggressive(Signal.BUY, 10, True)  # synthetic short
    sig, qty, inv = s.decide_aggressive(0.0)
    assert sig == Signal.SELL
    assert qty == 10
    assert inv is True


def test_decide_aggressive_pyramids_within_cap():
    s = EtfInavArbitrage(qty_per_step=5, max_position=50)
    s.apply_aggressive(Signal.BUY, 10, False)  # already long 10 (from 60bps move)
    sig, qty, inv = s.decide_aggressive(-0.012)  # widened to 120bps → wants 20 total
    assert sig == Signal.BUY
    assert qty == 10  # add 10 more to reach 20
    assert inv is False


def test_decide_aggressive_no_pyramid_when_at_target():
    s = EtfInavArbitrage(qty_per_step=5, max_position=50)
    s.apply_aggressive(Signal.BUY, 10, False)  # long 10 from 60bps
    sig, qty, inv = s.decide_aggressive(-0.006)  # still 60bps → target still 10
    assert sig == Signal.HOLD


def test_decide_aggressive_respects_max_position_cap():
    s = EtfInavArbitrage(qty_per_step=5, max_position=50)
    s.apply_aggressive(Signal.BUY, 50, False)  # already at cap
    sig, qty, inv = s.decide_aggressive(-0.030)  # 300bps → would want 50
    assert sig == Signal.HOLD  # no headroom


def test_decide_with_fair_blend_zero_matches_aggressive():
    """fair_blend=0 → fair_dev 무시, basket only."""
    s = EtfInavArbitrage(qty_per_step=5)
    sig_a, qty_a, inv_a = s.decide_aggressive(-0.006)
    s2 = EtfInavArbitrage(qty_per_step=5)
    sig_b, qty_b, inv_b = s2.decide_with_fair(
        dev_basket=-0.006, fair_dev=999.0, fair_blend=0.0,
    )
    assert (sig_a, qty_a, inv_a) == (sig_b, qty_b, inv_b)


def test_decide_with_fair_blend_full_uses_only_fair():
    """fair_blend=1 → basket 무시, fair_dev 만."""
    s = EtfInavArbitrage(qty_per_step=5)
    sig_a, qty_a, inv_a = s.decide_aggressive(-0.006)
    s2 = EtfInavArbitrage(qty_per_step=5)
    sig_b, qty_b, inv_b = s2.decide_with_fair(
        dev_basket=999.0, fair_dev=-0.006, fair_blend=1.0,
    )
    assert (sig_a, qty_a, inv_a) == (sig_b, qty_b, inv_b)


def test_decide_with_fair_blend_combines():
    """fair_blend=0.5 → 두 dev 평균."""
    s = EtfInavArbitrage(qty_per_step=5)
    # basket=-0.010, fair=-0.002 → combined=-0.006 → 2 step → 10주
    sig, qty, inv = s.decide_with_fair(
        dev_basket=-0.010, fair_dev=-0.002, fair_blend=0.5,
    )
    assert sig == Signal.BUY
    assert qty == 10  # |-0.006|/0.003 = 2 steps × 5
    assert inv is False


def test_apply_aggressive_inverse_position_tracked_separately():
    s = EtfInavArbitrage(qty_per_step=5)
    s.apply_aggressive(Signal.BUY, 5, False)
    s.apply_aggressive(Signal.BUY, 7, True)
    assert s.position == 5
    assert s.inverse_position == 7
    s.apply_aggressive(Signal.SELL, 5, False)
    assert s.position == 0
    assert s.inverse_position == 7


def test_risk_limits_block_outside_hours():
    now = datetime(2026, 1, 1, 8, 30)  # before 9am
    ok, reason = check(RiskState(), RiskLimits(), now=now)
    assert not ok


def test_risk_block_daily_loss():
    state = RiskState(equity_open=100, equity_now=97, position_value=0)
    now = datetime(2026, 1, 1, 11, 0)  # within hours
    ok, reason = check(state, RiskLimits(), now=now)
    assert not ok


def test_as_quote_symmetric_when_flat():
    cfg = ASConfig(gamma=0.1, sigma=0.01, k=1.5, tick_size=0.0)
    state = ASState(inventory=0)
    q = compute_quote(state, mid=100.0, t_remaining=1.0, cfg=cfg)
    assert abs((q.bid + q.ask) / 2 - q.reservation) < 1e-9
    assert abs(q.reservation - 100.0) < 1e-9
    assert q.ask > q.bid
    assert q.spread > 0


def test_as_quote_skews_with_inventory():
    cfg = ASConfig(gamma=0.5, sigma=0.02, k=1.5, tick_size=0.0)
    long_state = ASState(inventory=10)
    short_state = ASState(inventory=-10)
    q_long = compute_quote(long_state, mid=100.0, t_remaining=1.0, cfg=cfg)
    q_short = compute_quote(short_state, mid=100.0, t_remaining=1.0, cfg=cfg)
    assert q_long.reservation < 100.0
    assert q_short.reservation > 100.0
    assert q_long.reservation < q_short.reservation


def test_as_spread_shrinks_near_session_end():
    cfg = ASConfig(gamma=0.5, sigma=0.05, k=1.5, tick_size=0.0)
    state = ASState(inventory=0)
    q_start = compute_quote(state, mid=100.0, t_remaining=1.0, cfg=cfg)
    q_end = compute_quote(state, mid=100.0, t_remaining=0.0, cfg=cfg)
    assert q_end.spread < q_start.spread


def test_as_fill_check_sides():
    cfg = ASConfig(tick_size=0.0)
    state = ASState(inventory=0)
    q = compute_quote(state, mid=100.0, t_remaining=1.0, cfg=cfg)
    assert fill_check(q, mid_next=q.bid - 0.01, cfg=cfg) == "bid"
    assert fill_check(q, mid_next=q.ask + 0.01, cfg=cfg) == "ask"
    assert fill_check(q, mid_next=q.reservation, cfg=cfg) == "none"


def test_as_update_state_buy_then_sell_realizes_pnl():
    cfg = ASConfig(tick_size=0.0)
    state = ASState()
    q_buy = Quote(bid=99.0, ask=101.0, reservation=100.0, spread=2.0,
                        inventory=0, t_remaining=1.0)
    update_state(state, "bid", q_buy, qty=1)
    assert state.inventory == 1
    assert state.cash == -99.0
    q_sell = Quote(bid=100.5, ask=102.5, reservation=101.5, spread=2.0,
                         inventory=1, t_remaining=0.5)
    update_state(state, "ask", q_sell, qty=1)
    assert state.inventory == 0
    assert state.cash == -99.0 + 102.5
    assert mark_to_market(state, mid=100.0) == state.cash
