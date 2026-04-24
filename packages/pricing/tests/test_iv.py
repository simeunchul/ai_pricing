from pricing.bsm import BSMInputs, call_price
from pricing.iv import implied_vol


def test_iv_roundtrip():
    i = BSMInputs(100, 100, 1, 0.03, 0, 0.2)
    p = call_price(i)
    iv = implied_vol(p, i, opt="call")
    assert abs(iv - 0.2) < 1e-6


def test_iv_different_strikes():
    for K in (80, 90, 100, 110, 120):
        i = BSMInputs(100, K, 0.5, 0.03, 0, 0.25)
        p = call_price(i)
        iv = implied_vol(p, i, opt="call")
        assert abs(iv - 0.25) < 1e-6
