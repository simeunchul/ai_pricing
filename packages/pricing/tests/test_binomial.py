from pricing.bsm import BSMInputs, call_price
from pricing.binomial import binomial_american, binomial_european


def test_binomial_converges_to_bsm():
    i = BSMInputs(100, 100, 1, 0.03, 0, 0.2)
    bsm = call_price(i)
    bin_price = binomial_european(i, N=2000, opt="call")
    assert abs(bin_price - bsm) / bsm < 1e-3


def test_american_call_no_div_equals_european():
    """No early exercise premium for American call w/o dividends."""
    i = BSMInputs(100, 100, 1, 0.05, 0, 0.25)
    eu = binomial_european(i, N=1000, opt="call")
    am = binomial_american(i, N=1000, opt="call")
    assert abs(eu - am) / eu < 1e-3


def test_american_put_exceeds_european():
    i = BSMInputs(100, 110, 1, 0.05, 0, 0.3)
    eu = binomial_european(i, N=500, opt="put")
    am = binomial_american(i, N=500, opt="put")
    assert am >= eu - 1e-6
    assert am > eu  # should be strictly greater for ITM put
