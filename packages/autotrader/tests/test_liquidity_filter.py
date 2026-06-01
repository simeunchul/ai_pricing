"""Liquidity filter (Phase 1 + Phase 2) 단위 테스트."""

from __future__ import annotations

from autotrader.market.liquidity_filter import (
    check_trading_value, check_spread, parse_asking_price,
    evaluate_liquidity,
    MIN_TRADING_VALUE_KRW, MAX_SPREAD_BPS,
)


# ---------- Phase 1: 거래대금 필터 ----------

def test_trading_value_pass_large_cap():
    # 삼성전자급: 거래대금 = 75000 × 20_000_000주 = 1.5조
    ok, tv, reason = check_trading_value("005930", 75000, 20_000_000)
    assert ok
    assert tv == 1_500_000_000_000
    assert "OK" in reason


def test_trading_value_fail_small_cap():
    # 소형주: 5000원 × 50_000주 = 2.5억 < 5억 임계
    ok, tv, reason = check_trading_value("XXXXXX", 5000, 50_000)
    assert not ok
    assert tv == 250_000_000
    assert "부족" in reason


def test_trading_value_edge_at_threshold():
    # 정확히 5억 — 통과
    ok, tv, reason = check_trading_value("XXX", 10000, 50_000)  # 5억 정확히
    assert ok


def test_trading_value_custom_threshold():
    # 10억 임계로 키우면 5억 종목은 탈락
    ok, _, _ = check_trading_value("XXX", 10000, 50_000, min_krw=1_000_000_000)
    assert not ok


# ---------- Phase 2: 호가 스프레드 ----------

SAMPLE_GOOD_SPREAD = {
    "output1": {
        "askp1": "75100", "bidp1": "75000",   # spread 100 / mid 75050 = 13.32 bps
        "total_askp_rsqn": "120000", "total_bidp_rsqn": "150000",
    }
}

SAMPLE_BAD_SPREAD = {
    "output1": {
        "askp1": "5200", "bidp1": "5000",     # spread 200 / mid 5100 = 392 bps
        "total_askp_rsqn": "1000", "total_bidp_rsqn": "800",
    }
}

SAMPLE_INVALID = {"output1": {}}


def test_parse_asking_price_basic():
    p = parse_asking_price(SAMPLE_GOOD_SPREAD)
    assert p["ask"] == 75100
    assert p["bid"] == 75000
    assert p["mid"] == 75050.0
    assert p["spread"] == 100
    assert abs(p["spread_bps"] - 13.32) < 0.5


def test_parse_asking_price_invalid_returns_empty():
    p = parse_asking_price(SAMPLE_INVALID)
    assert p == {}
    p2 = parse_asking_price({})
    assert p2 == {}


def test_check_spread_pass_tight():
    ok, bps, reason = check_spread("005930", SAMPLE_GOOD_SPREAD)
    assert ok
    assert bps < 30
    assert "OK" in reason


def test_check_spread_fail_wide():
    ok, bps, reason = check_spread("XXX", SAMPLE_BAD_SPREAD)
    assert not ok
    assert bps > 30
    assert "과대" in reason


def test_check_spread_invalid_payload():
    ok, bps, reason = check_spread("XXX", SAMPLE_INVALID)
    assert not ok
    assert bps is None


# ---------- 통합 evaluate_liquidity ----------

def test_evaluate_phase1_only_pass():
    # asking_payload=None → Phase 1 만
    chk = evaluate_liquidity("005930", price=75000, acml_vol=20_000_000)
    assert chk.passed
    assert chk.spread_bps is None      # Phase 2 미실행
    assert chk.trading_value > 1e12


def test_evaluate_phase1_fail_short_circuits():
    # 거래대금 부족이면 Phase 2 실행 안 함
    chk = evaluate_liquidity(
        "XXX", price=5000, acml_vol=10_000,
        asking_payload=SAMPLE_GOOD_SPREAD,    # spread 좋아도
    )
    assert not chk.passed
    assert "부족" in chk.reason


def test_evaluate_phase1_pass_phase2_pass():
    chk = evaluate_liquidity(
        "005930", price=75000, acml_vol=20_000_000,
        asking_payload=SAMPLE_GOOD_SPREAD,
    )
    assert chk.passed
    assert chk.spread_bps is not None
    assert chk.spread_bps < 30


def test_evaluate_phase1_pass_phase2_fail():
    # 거래대금 OK 인데 호가 spread 너무 넓음
    chk = evaluate_liquidity(
        "XXX", price=10000, acml_vol=100_000,   # 10억 — Phase 1 OK
        asking_payload=SAMPLE_BAD_SPREAD,        # 392bps — Phase 2 fail
    )
    assert not chk.passed
    assert chk.spread_bps > 30
    assert "과대" in chk.reason


def test_evaluate_custom_thresholds():
    # 매우 빡빡하게 (거래대금 100억, spread 10bps)
    chk = evaluate_liquidity(
        "005930", price=75000, acml_vol=20_000_000,
        asking_payload=SAMPLE_GOOD_SPREAD,
        min_trading_value_krw=10_000_000_000,    # 100억
        max_spread_bps=10,                         # 10bps
    )
    # 거래대금 1.5조 > 100억 OK
    # spread 13bps > 10bps fail
    assert not chk.passed
    assert "과대" in chk.reason


# ---------- 상수 sanity ----------

def test_default_constants_reasonable():
    assert MIN_TRADING_VALUE_KRW == 500_000_000   # 5억
    assert MAX_SPREAD_BPS == 30                     # 0.3%
