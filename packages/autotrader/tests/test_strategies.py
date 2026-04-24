from autotrader.market.inav import InavEstimator, deviation
from autotrader.strategies.etf_inav_arb import EtfInavArbitrage, Signal
from autotrader.risk.limits import RiskLimits, RiskState, check
from datetime import datetime


def test_inav_basic():
    est = InavEstimator(constituents={"A": 0.5, "B": 0.5})
    assert est.estimate({"A": 100, "B": 200}) == 150.0


def test_deviation_sign():
    assert deviation(101, 100) > 0
    assert deviation(99, 100) < 0
    assert deviation(100, 100) == 0


def test_strategy_flat_to_buy_when_cheap():
    s = EtfInavArbitrage(enter_threshold=0.003)
    sig = s.decide(-0.005)
    assert sig == Signal.BUY


def test_strategy_exits_on_convergence():
    s = EtfInavArbitrage(enter_threshold=0.003, exit_threshold=0.0005)
    s.apply(Signal.BUY)  # long 1
    sig = s.decide(0.0)
    assert sig == Signal.SELL


def test_risk_limits_block_outside_hours():
    now = datetime(2026, 1, 1, 8, 30)  # before 9am
    ok, reason = check(RiskState(), RiskLimits(), now=now)
    assert not ok


def test_risk_block_daily_loss():
    state = RiskState(equity_open=100, equity_now=97, position_value=0)
    now = datetime(2026, 1, 1, 11, 0)  # within hours
    ok, reason = check(state, RiskLimits(), now=now)
    assert not ok
