from pricing.bsm import BSMInputs, call_price
from pricing.mc.engine import mc_price
from pricing.payoffs import call_payoff


def test_mc_converges_to_bsm():
    i = BSMInputs(100, 100, 1, 0.03, 0, 0.2)
    bsm = call_price(i)
    res = mc_price(i, call_payoff(100), n_paths=200_000, n_steps=50, seed=42)
    # within 3 stderr
    assert abs(res.price - bsm) < 3 * res.stderr + 0.05
