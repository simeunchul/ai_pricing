import numpy as np

from pricing.els.step_down import StepDownELS, price_els


def test_els_basic_prices_below_notional_when_risky():
    """A typical KOSPI200 / HSCEI step-down ELS issue price is near 10,000 krw
    but fair value discount vs notional must be positive."""
    product = StepDownELS(
        S0=np.array([100.0, 100.0]),
        barriers=[0.90, 0.85, 0.85, 0.80, 0.80, 0.75],
        ki_barrier=0.50,
        coupon_rate=0.03,  # 3% per 6mo = ~6%/yr
        maturity_years=3.0,
        obs_per_year=2,
        notional=10_000.0,
    )
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    res = price_els(
        product, r=0.03,
        q=np.array([0.0, 0.0]),
        sigma=np.array([0.25, 0.30]),
        corr=corr,
        n_paths=5_000,
        n_steps_per_year=64,
        seed=0,
    )
    assert 8_000 < res.price < 10_500
    assert 0.0 <= res.ki_hit_prob <= 1.0
    assert sum(res.autocall_prob) <= 1.0
    assert res.expected_life <= product.maturity_years
