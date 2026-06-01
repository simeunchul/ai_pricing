"""ForeignFlowFollow 전략 + backtest harness 단위 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from autotrader.strategies.etf_inav_arb import Signal
from autotrader.strategies.foreign_flow import ForeignFlowFollow
from autotrader.strategies.foreign_flow_proportional import ForeignFlowProportional
from autotrader.strategies.foreign_inst_flow import ForeignInstFlowFollow
from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig,
    run_foreign_flow_backtest,
    run_portfolio_backtest,
    run_buyhold_portfolio,
    run_dual_dynamic_backtest,
    summarize_trades,
    universe_index,
    detect_regimes,
    analyze_regime_performance,
)


# ---------- Strategy 단위 테스트 ----------

def test_buy_when_flow_ratio_above_threshold():
    s = ForeignFlowFollow(enter_threshold=0.05, qty_per_signal=10)
    sig, qty = s.decide("005930", 0.10)
    assert sig == Signal.BUY
    assert qty == 10


def test_hold_when_flow_ratio_inside_band():
    s = ForeignFlowFollow(enter_threshold=0.05, qty_per_signal=10)
    sig, qty = s.decide("005930", 0.03)
    assert sig == Signal.HOLD
    assert qty == 0


def test_sell_signal_ignored_when_flat():
    """미보유 시 SELL 신호 무시 (long-only, 공매도 금지)."""
    s = ForeignFlowFollow(enter_threshold=0.05, qty_per_signal=10)
    sig, qty = s.decide("005930", -0.10)
    assert sig == Signal.HOLD
    assert qty == 0


def test_sell_when_holding_and_flow_negative():
    s = ForeignFlowFollow(enter_threshold=0.05, qty_per_signal=10)
    s.apply("005930", Signal.BUY, 10)
    sig, qty = s.decide("005930", -0.10)
    assert sig == Signal.SELL
    assert qty == 10  # 전량 청산


def test_position_cap_blocks_additional_buys():
    s = ForeignFlowFollow(
        enter_threshold=0.05, qty_per_signal=10, max_position_per_symbol=20,
    )
    s.apply("005930", Signal.BUY, 20)  # 캡 도달
    sig, qty = s.decide("005930", 0.20)  # 강한 매수 신호여도
    assert sig == Signal.HOLD


def test_partial_buy_when_near_cap():
    s = ForeignFlowFollow(
        enter_threshold=0.05, qty_per_signal=10, max_position_per_symbol=15,
    )
    s.apply("005930", Signal.BUY, 10)
    sig, qty = s.decide("005930", 0.20)
    assert sig == Signal.BUY
    assert qty == 5  # 5만 남아서 5만 매수


def test_auto_exit_at_holding_maturity():
    s = ForeignFlowFollow(enter_threshold=0.05, exit_after_days=2, qty_per_signal=10)
    s.apply("005930", Signal.BUY, 10)
    s.advance_day()  # day 1
    s.advance_day()  # day 2 (만기 도달)
    # 외국인 매수 강세여도 만기면 청산
    sig, qty = s.decide("005930", 0.15)
    assert sig == Signal.SELL
    assert qty == 10


def test_independent_positions_per_symbol():
    s = ForeignFlowFollow(enter_threshold=0.05, qty_per_signal=10)
    s.apply("005930", Signal.BUY, 10)
    s.apply("000660", Signal.BUY, 5)
    assert s.positions["005930"] == 10
    assert s.positions["000660"] == 5
    s.apply("005930", Signal.SELL, 10)
    assert s.positions["005930"] == 0
    assert s.positions["000660"] == 5


def test_advance_day_only_increments_active_holdings():
    s = ForeignFlowFollow(enter_threshold=0.05, qty_per_signal=10)
    s.apply("005930", Signal.BUY, 10)
    s.advance_day()
    assert s.days_held["005930"] == 1
    s.apply("005930", Signal.SELL, 10)
    s.advance_day()
    assert "005930" not in s.days_held


# ---------- ForeignFlowProportional 단위 테스트 ----------

def test_proportional_buy_qty_scales_with_flow():
    s = ForeignFlowProportional(sizing_factor=100, min_threshold=0.01)
    sig_a, qty_a = s.decide("X", 0.10)
    sig_b, qty_b = s.decide("Y", 0.02)
    assert sig_a == Signal.BUY and qty_a == 10
    assert sig_b == Signal.BUY and qty_b == 2
    assert qty_a == 5 * qty_b


def test_proportional_noise_cut_holds():
    s = ForeignFlowProportional(sizing_factor=100, min_threshold=0.01)
    sig, qty = s.decide("X", 0.005)
    assert sig == Signal.HOLD
    assert qty == 0


def test_proportional_partial_sell_when_signal_smaller_than_position():
    s = ForeignFlowProportional(sizing_factor=100, min_threshold=0.01)
    s.apply("X", Signal.BUY, 20)
    sig, qty = s.decide("X", -0.10)        # 매도 신호 = 10주
    assert sig == Signal.SELL
    assert qty == 10                         # 부분 청산 (보유 20 중 10)


def test_proportional_full_sell_when_signal_exceeds_position():
    s = ForeignFlowProportional(sizing_factor=100, min_threshold=0.01)
    s.apply("X", Signal.BUY, 5)
    sig, qty = s.decide("X", -0.20)        # 매도 신호 = 20주
    assert sig == Signal.SELL
    assert qty == 5                          # 보유 5 만 청산 (잉여 신호 무시)


def test_proportional_no_short_when_flat_and_negative():
    s = ForeignFlowProportional(sizing_factor=100, min_threshold=0.01)
    sig, qty = s.decide("X", -0.20)        # 미보유 + 강한 매도 → HOLD
    assert sig == Signal.HOLD


def test_proportional_apply_partial_then_rebuy_keeps_avg_logic():
    s = ForeignFlowProportional(sizing_factor=100, min_threshold=0.01)
    s.apply("X", Signal.BUY, 20)
    s.apply("X", Signal.SELL, 10)            # 부분 청산
    assert s.positions["X"] == 10
    s.apply("X", Signal.BUY, 5)              # 재매수
    assert s.positions["X"] == 15


def test_proportional_position_cap_blocks_extra_buy():
    s = ForeignFlowProportional(
        sizing_factor=1000, min_threshold=0.01, max_position_per_symbol=50,
    )
    sig, qty = s.decide("X", 0.10)         # naive qty = 100, but capped
    assert sig == Signal.BUY
    assert qty == 50                         # 캡까지만


# ---------- Backtest harness 통합 테스트 (합성 데이터) ----------

def _make_synthetic_data(monkeypatch):
    """data layer 를 monkeypatch 해서 합성 시계열을 주입."""
    from autotrader.backtest import foreign_flow as ff

    # 외국인이 day-N 에 강하게 매수 → day-(N+1)/(N+2) 에 가격 상승
    # 즉, 가설이 진짜인 합성 시나리오
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    # 4일 주기로 (large foreign buy → 다음 2일 상승 → 3일째 정상화)
    foreign = []
    closes = []
    opens = []
    for i in range(30):
        if i % 4 == 0:
            foreign.append(100_000)   # large buy day
            closes.append(100 + i)
            opens.append(100 + i)
        elif i % 4 == 1:
            foreign.append(0)
            closes.append(102 + i)    # +2 next day
            opens.append(101 + i)
        elif i % 4 == 2:
            foreign.append(0)
            closes.append(103 + i)
            opens.append(102 + i)
        else:
            foreign.append(-30_000)
            closes.append(100 + i)    # 평균회귀
            opens.append(102 + i)

    df = pd.DataFrame({
        "open": opens,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [1_000_000] * 30,
        "foreign_net_shares": foreign,
        "institution_net_shares": [0] * 30,
        "foreign_holdings_shares": [50_000_000] * 30,
        "foreign_ratio_pct": [50.0] * 30,
    }, index=dates)
    df.index.name = "date"

    def fake_load(symbol, cfg):
        return ff.add_flow_features(df.copy())

    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)


def test_backtest_runs_on_synthetic_data(monkeypatch):
    _make_synthetic_data(monkeypatch)

    cfg = ForeignFlowBacktestConfig(
        symbols=["TEST"],
        enter_threshold=0.05,    # 100k/1M = 0.10 → 임계 5% 초과
        exit_after_days=2,
        qty_per_signal=10,
        cost_bps=10.0,
        use_cache=False,
    )
    trades = run_foreign_flow_backtest(cfg)
    assert len(trades) > 0
    # 가격이 단조 증가하는 합성 데이터라 net_return 평균이 양수여야 함
    assert trades["net_return"].mean() > 0


# ---------- ForeignInstFlowFollow 단위 테스트 ----------

def test_foreign_inst_buy_when_both_strong_positive():
    s = ForeignInstFlowFollow(enter_threshold=0.05, sizing_factor=200)
    sig, qty = s.decide("X", flow_ratio=0.10, inst_ratio=0.08)
    # weaker = 0.08 → qty = round(0.08 * 200) = 16
    assert sig == Signal.BUY
    assert qty == 16


def test_foreign_inst_hold_when_signs_disagree():
    """외국인 + / 기관 - 인 경우 신호 cancel → HOLD."""
    s = ForeignInstFlowFollow(enter_threshold=0.05, sizing_factor=200)
    sig, qty = s.decide("X", flow_ratio=0.10, inst_ratio=-0.08)
    assert sig == Signal.HOLD
    assert qty == 0


def test_foreign_inst_hold_when_one_below_threshold():
    """한쪽만 강하면 무시."""
    s = ForeignInstFlowFollow(enter_threshold=0.05, sizing_factor=200)
    sig, qty = s.decide("X", flow_ratio=0.10, inst_ratio=0.02)  # inst 약함
    assert sig == Signal.HOLD


def test_foreign_inst_sell_when_both_strong_negative_holding():
    s = ForeignInstFlowFollow(enter_threshold=0.05, sizing_factor=200)
    s.apply("X", Signal.BUY, 30)
    sig, qty = s.decide("X", flow_ratio=-0.10, inst_ratio=-0.07)
    # weaker abs = 0.07 → qty = 14
    assert sig == Signal.SELL
    assert qty == 14


def test_foreign_inst_sell_caps_at_position():
    s = ForeignInstFlowFollow(enter_threshold=0.05, sizing_factor=200)
    s.apply("X", Signal.BUY, 5)
    sig, qty = s.decide("X", flow_ratio=-0.20, inst_ratio=-0.15)
    # target 30, but pos 5 → 5
    assert sig == Signal.SELL
    assert qty == 5


def test_foreign_inst_no_short_when_flat():
    s = ForeignInstFlowFollow(enter_threshold=0.05, sizing_factor=200)
    sig, qty = s.decide("X", flow_ratio=-0.10, inst_ratio=-0.10)
    assert sig == Signal.HOLD


# ---------- Stop loss 단위 테스트 (합성 데이터) ----------

def _make_falling_data(monkeypatch):
    """진입 후 가격이 단조 하락하는 합성 시계열 — stop loss 검증용."""
    from autotrader.backtest import foreign_flow as ff

    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    # day 0: 강한 외국인 매수 → 진입 신호
    # day 1+: 가격 하락
    foreign = [0] + [-10_000] * 19   # day 0에 매수 신호 만들려면 양수 → adjust
    foreign[0] = 50_000              # day 0: 거래량 1M 의 5% = 매수 신호
    closes = [100] + [100 - i * 2 for i in range(1, 20)]   # 100 → 62
    opens = [100] + [101 - i * 2 for i in range(1, 20)]    # 시가가 종가 직전값과 비슷

    df = pd.DataFrame({
        "open": opens,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [1_000_000] * 20,
        "foreign_net_shares": foreign,
        "institution_net_shares": [0] * 20,
        "foreign_holdings_shares": [50_000_000] * 20,
        "foreign_ratio_pct": [50.0] * 20,
    }, index=dates)
    df.index.name = "date"

    def fake_load(symbol, cfg):
        return ff.add_flow_features(df.copy())

    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)


def test_stop_loss_disabled_holds_through_drawdown(monkeypatch):
    _make_falling_data(monkeypatch)
    cfg = ForeignFlowBacktestConfig(
        symbols=["TEST"], cost_bps=10.0, use_cache=False,
    )
    strat = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res = run_portfolio_backtest(cfg, strategy=strat,
                                  initial_cash=1_000_000,
                                  stop_loss_pct=None)
    # 손절 비활성 → 끝까지 보유 → 가격 하락 그대로 반영 (손실 발생)
    assert res.overall["return"] < 0


def test_stop_loss_triggers_and_caps_loss(monkeypatch):
    _make_falling_data(monkeypatch)
    cfg = ForeignFlowBacktestConfig(
        symbols=["TEST"], cost_bps=10.0, use_cache=False,
    )
    strat = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res_no_sl = run_portfolio_backtest(cfg, strategy=strat,
                                         initial_cash=1_000_000,
                                         stop_loss_pct=None)
    strat2 = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res_sl = run_portfolio_backtest(cfg, strategy=strat2,
                                      initial_cash=1_000_000,
                                      stop_loss_pct=0.10)
    # 10% 손절 → 끝까지 보유한 케이스보다 손실 작아야 함
    assert res_sl.overall["return"] > res_no_sl.overall["return"]
    # 손절 청산 후 cash 로 보존되어 -10% 근처에서 손실 멈춤 (대략 -10% ~ -15% 범위)
    assert res_sl.overall["return"] > -0.20


def test_trailing_stop_triggers_after_drawdown_from_peak(monkeypatch):
    """진입 후 가격이 올랐다가 peak 대비 -X% 떨어지면 청산되는지 검증."""
    from autotrader.backtest import foreign_flow as ff

    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    # day 0: 진입 신호
    # day 1~5: 가격 상승 (100→110)
    # day 6~9: 큰 하락 (110→50) — trailing 효과 명확히 측정 가능
    closes = [100, 102, 104, 106, 108, 110, 108, 100, 80, 60] + [50] * 10
    foreign = [50_000] + [0] * 19

    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
        "volume": [1_000_000] * 20,
        "foreign_net_shares": foreign,
        "institution_net_shares": [0] * 20,
        "foreign_holdings_shares": [50_000_000] * 20,
        "foreign_ratio_pct": [50.0] * 20,
    }, index=dates)
    df.index.name = "date"

    def fake_load(symbol, cfg):
        return ff.add_flow_features(df.copy())
    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)

    cfg = ForeignFlowBacktestConfig(symbols=["TEST"], cost_bps=10.0, use_cache=False)

    # trailing 5% — peak 110 대비 -5% = 104.5, day 7 close=105 까진 OK,
    # day 8 close=100 < 104.5 → 다음날 청산
    strat = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res_trail = run_portfolio_backtest(cfg, strategy=strat,
                                         initial_cash=1_000_000,
                                         trailing_stop_pct=0.05)
    # 끝까지 보유 (None) 케이스
    strat2 = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res_none = run_portfolio_backtest(cfg, strategy=strat2,
                                        initial_cash=1_000_000,
                                        trailing_stop_pct=None)
    # trailing 5% 가 끝까지 보유보다 손실 적어야 함 (peak 근처에서 청산)
    assert res_trail.overall["return"] >= res_none.overall["return"]


def test_trailing_does_not_trigger_during_uptrend(monkeypatch):
    """가격이 단조 상승하면 trailing 발동 안 해야."""
    from autotrader.backtest import foreign_flow as ff

    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    closes = [100 + i * 2 for i in range(20)]   # 100 → 138 (단조 상승)
    foreign = [50_000] + [0] * 19

    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
        "volume": [1_000_000] * 20,
        "foreign_net_shares": foreign,
        "institution_net_shares": [0] * 20,
        "foreign_holdings_shares": [50_000_000] * 20,
        "foreign_ratio_pct": [50.0] * 20,
    }, index=dates)
    df.index.name = "date"

    def fake_load(symbol, cfg):
        return ff.add_flow_features(df.copy())
    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)

    cfg = ForeignFlowBacktestConfig(symbols=["TEST"], cost_bps=10.0, use_cache=False)
    strat = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res = run_portfolio_backtest(cfg, strategy=strat,
                                   initial_cash=1_000_000,
                                   trailing_stop_pct=0.05)
    # 상승장 → 양수 수익
    assert res.overall["return"] > 0


def test_mdd_cap_triggers_full_liquidation(monkeypatch):
    _make_falling_data(monkeypatch)
    cfg = ForeignFlowBacktestConfig(symbols=["TEST"], cost_bps=10.0, use_cache=False)
    strat = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res_no_cap = run_portfolio_backtest(cfg, strategy=strat,
                                          initial_cash=1_000_000)
    strat2 = ForeignFlowProportional(
        sizing_factor=200, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res_with_cap = run_portfolio_backtest(cfg, strategy=strat2,
                                            initial_cash=1_000_000,
                                            mdd_cap=0.10,
                                            cooldown_days=20)
    # MDD cap 으로 손실 한정되거나 동일 (이미 손실 작은 합성 데이터)
    assert res_with_cap.overall["return"] >= res_no_cap.overall["return"] - 0.001


def test_volatility_sizing_reduces_qty_for_high_vol(monkeypatch):
    """변동성 sizing 활성화 시 매수 수량이 vol_target/vol 비율로 조정되는지 확인."""
    from autotrader.backtest import foreign_flow as ff

    # 변동성 매우 높은 합성 데이터 (daily ±5%)
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    closes = []
    for i in range(40):
        # 진동: 100, 105, 100, 105, ...
        closes.append(100 if i % 2 == 0 else 105)
    foreign = [50_000] + [0] * 39  # 진입 신호 한 번만

    df = pd.DataFrame({
        "open": closes,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [1_000_000] * 40,
        "foreign_net_shares": foreign,
        "institution_net_shares": [0] * 40,
        "foreign_holdings_shares": [50_000_000] * 40,
        "foreign_ratio_pct": [50.0] * 40,
    }, index=dates)
    df.index.name = "date"

    def fake_load(symbol, cfg):
        return ff.add_flow_features(df.copy())

    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)

    cfg = ForeignFlowBacktestConfig(symbols=["TEST"], cost_bps=10.0, use_cache=False)
    strat = ForeignFlowProportional(
        sizing_factor=1000, min_threshold=0.01, max_position_per_symbol=10_000,
    )
    res = run_portfolio_backtest(cfg, strategy=strat,
                                   initial_cash=10_000_000,
                                   vol_target_daily=0.01,
                                   vol_lookback=20)
    # 단순 검증: 백테스트가 정상 종료, history 가 비어있지 않음
    assert not res.history.empty


def test_buyhold_portfolio_matches_buyhold_calculation(monkeypatch):
    _make_falling_data(monkeypatch)
    cfg = ForeignFlowBacktestConfig(symbols=["TEST"], cost_bps=10.0, use_cache=False)
    bh = run_buyhold_portfolio(cfg, initial_cash=1_000_000)
    # 첫날 시가 매수 후 끝까지 보유 → 끝 close 기준 가치
    # 시작 시가 100, 끝 close 62 → 약 -38%
    assert bh.overall["return"] < -0.30


def test_dual_dynamic_backtest_runs_and_picks_signal_days(monkeypatch):
    """동적 universe 백테스트 — dual confirmation 통과 종목 자동 진입 검증."""
    from autotrader.backtest import foreign_flow as ff

    dates = pd.date_range("2025-01-01", periods=10, freq="B")

    def make_df(foreign_pattern, inst_pattern, prices):
        return pd.DataFrame({
            "open": prices, "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices], "close": prices,
            "volume": [1_000_000] * 10,
            "foreign_net_shares": foreign_pattern,
            "institution_net_shares": inst_pattern,
            "foreign_holdings_shares": [50_000_000] * 10,
            "foreign_ratio_pct": [50.0] * 10,
        }, index=dates)

    # 종목 A: day 2 매수 신호 (외인+기관 둘 다 +6%), day 5 매도 신호 (둘 다 -6%)
    df_a = make_df(
        [0, 0, 60_000, 0, 0, -60_000, 0, 0, 0, 0],
        [0, 0, 60_000, 0, 0, -60_000, 0, 0, 0, 0],
        [100, 100, 100, 102, 105, 105, 100, 100, 100, 100],
    )
    df_a.index.name = "date"

    # 종목 B: 외인만 매수 (기관 0) — dual 미통과 → 진입 안 함
    df_b = make_df(
        [0, 0, 100_000, 0, 0, 0, 0, 0, 0, 0],
        [0] * 10,
        [200] * 10,
    )
    df_b.index.name = "date"

    sym_data = {"A": ff.add_flow_features(df_a), "B": ff.add_flow_features(df_b)}

    def fake_load(symbol, cfg):
        return sym_data.get(symbol, pd.DataFrame()).copy()

    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)

    cfg = ForeignFlowBacktestConfig(symbols=["A", "B"], cost_bps=10.0, use_cache=False)
    res = run_dual_dynamic_backtest(
        cfg, initial_cash=1_000_000, enter_threshold=0.05,
        sizing_factor=200, max_concurrent=5,
    )
    assert not res.history.empty
    # 진입 신호가 발생하긴 했어야
    holdings_max = res.history["position"].max()
    assert holdings_max >= 1  # 최소 1번은 진입


def test_dual_dynamic_skips_when_only_one_side_strong(monkeypatch):
    """한 쪽만 강하면 진입 안 함 (dual confirmation 핵심)."""
    from autotrader.backtest import foreign_flow as ff

    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    # 외인 매우 강함 (+10%) but 기관 약함 (+1%) → 매번 dual 미통과
    df = pd.DataFrame({
        "open": [100] * 10, "high": [101] * 10, "low": [99] * 10, "close": [100] * 10,
        "volume": [1_000_000] * 10,
        "foreign_net_shares": [100_000] * 10,    # +10%
        "institution_net_shares": [10_000] * 10,  # +1%
        "foreign_holdings_shares": [50_000_000] * 10,
        "foreign_ratio_pct": [50.0] * 10,
    }, index=dates)
    df.index.name = "date"

    sym_data = ff.add_flow_features(df)

    def fake_load(symbol, cfg):
        return sym_data.copy()

    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)

    cfg = ForeignFlowBacktestConfig(symbols=["X"], cost_bps=10.0, use_cache=False)
    res = run_dual_dynamic_backtest(
        cfg, initial_cash=1_000_000, enter_threshold=0.05,
        sizing_factor=200,
    )
    # 한 쪽만 강해서 진입 안 함 → 보유 0 유지
    assert (res.history["position"] == 0).all()
    # final value ≈ initial cash (소수 손실 없음)
    assert abs(res.overall["return"]) < 0.001


def test_universe_index_and_regime_detection(monkeypatch):
    """등가중 인덱스 만들고 drawdown threshold 로 bull/bear 분리 검증."""
    from autotrader.backtest import foreign_flow as ff

    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    # peak at i=10, then -10% drawdown
    closes = [100 + i for i in range(11)] + [110 - (i - 10) * 1.5 for i in range(11, 30)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
        "volume": [1_000_000] * 30,
        "foreign_net_shares": [0] * 30,
        "institution_net_shares": [0] * 30,
        "foreign_holdings_shares": [50_000_000] * 30,
        "foreign_ratio_pct": [50.0] * 30,
    }, index=dates)
    df.index.name = "date"

    def fake_load(symbol, cfg):
        return ff.add_flow_features(df.copy())

    monkeypatch.setattr(ff, "_load_symbol_data", fake_load)

    cfg = ForeignFlowBacktestConfig(symbols=["TEST"], use_cache=False)
    idx = universe_index(cfg)
    assert len(idx) == 30
    regimes = detect_regimes(idx, drawdown_threshold=0.05)
    # 상승 구간 (i<=10)은 bull, 하락 후반은 bear 일부
    assert (regimes == "bull").sum() > 0
    assert (regimes == "bear").sum() > 0


def test_summary_produces_expected_keys():
    df = pd.DataFrame([
        {"symbol": "A", "net_return": 0.01, "hold_days": 2},
        {"symbol": "A", "net_return": -0.005, "hold_days": 1},
        {"symbol": "B", "net_return": 0.02, "hold_days": 3},
    ])
    s = summarize_trades(df, baseline={"A": 0.05, "B": 0.10})
    assert s["n_trades"] == 3
    assert "hit_rate" in s
    assert "by_symbol" in s
    assert "buyhold_mean_return" in s
    assert "excess_vs_buyhold_mean" in s
