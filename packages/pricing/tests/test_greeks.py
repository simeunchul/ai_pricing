from pricing.bsm import BSMInputs
from pricing.greeks.analytic import call_greeks, put_greeks
from pricing.greeks.bumping import bumping_greeks


def test_call_greeks_analytic_vs_bumping():
    i = BSMInputs(100, 100, 1, 0.03, 0.01, 0.2)
    a = call_greeks(i)
    b = bumping_greeks(i, opt="call")

    assert abs(a.delta - b.delta) < 1e-3
    assert abs(a.gamma - b.gamma) / max(abs(a.gamma), 1e-6) < 1e-2
    assert abs(a.vega - b.vega) / max(abs(a.vega), 1e-6) < 1e-2
    assert abs(a.rho - b.rho) / max(abs(a.rho), 1e-6) < 1e-2


def test_put_delta_range():
    i = BSMInputs(100, 100, 1, 0.03, 0, 0.2)
    g = put_greeks(i)
    assert -1.0 <= g.delta <= 0.0


def test_call_delta_range():
    i = BSMInputs(100, 100, 1, 0.03, 0, 0.2)
    g = call_greeks(i)
    assert 0.0 <= g.delta <= 1.0
