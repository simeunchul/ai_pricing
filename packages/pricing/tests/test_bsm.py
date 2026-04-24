import math

import pytest

from pricing.bsm import BSMInputs, call_price, put_price, forward_price


def test_atm_call_reasonable():
    p = call_price(BSMInputs(100, 100, 1, 0.03, 0, 0.2))
    assert 7.0 < p < 10.0  # Hull textbook ballpark


def test_put_call_parity():
    i = BSMInputs(100, 95, 0.5, 0.05, 0.02, 0.25)
    c = call_price(i)
    p = put_price(i)
    lhs = c - p
    rhs = i.S * math.exp(-i.q * i.T) - i.K * math.exp(-i.r * i.T)
    assert abs(lhs - rhs) < 1e-10


def test_zero_vol_call_is_intrinsic_discounted():
    i = BSMInputs(100, 90, 1, 0.05, 0, 1e-9)
    c = call_price(i)
    expected = max(i.S - i.K * math.exp(-i.r * i.T), 0.0) - 0  # at sigma=0, forward value
    # With sigma=0 the forward = S*exp((r-q)T) and payoff = (F-K)^+ discounted
    fwd = forward_price(i)
    expected = max(fwd - i.K, 0) * math.exp(-i.r * i.T)
    assert abs(c - expected) < 1e-3


def test_validation():
    with pytest.raises(ValueError):
        BSMInputs(100, 100, -1, 0.03, 0, 0.2)
    with pytest.raises(ValueError):
        BSMInputs(100, 100, 1, 0.03, 0, -0.2)
