import math

import pytest

from pricing.bsm import BSMInputs
from pricing.heston import HestonParams, heston_call_semi, heston_mc
from pricing.payoffs import call_payoff


def test_heston_semi_vs_mc():
    """Semi-analytic Heston and MC Heston should agree within a few bps on MC stderr."""
    pytest.importorskip("scipy")
    S, K, T, r = 100.0, 100.0, 1.0, 0.03
    p = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)

    semi = heston_call_semi(S, K, T, r, p)
    mc = heston_mc(
        BSMInputs(S, K, T, r, 0.0, math.sqrt(p.v0)),  # sigma field unused but required
        p,
        call_payoff(K),
        n_paths=40_000,
        n_steps=200,
        seed=0,
    )
    # 3 stderr + small buffer
    assert abs(semi - mc.price) < 3 * mc.stderr + 0.1


def test_heston_reduces_to_bsm_when_vol_const():
    """If kappa very large and theta=v0, vol should effectively stay constant."""
    p = HestonParams(kappa=100.0, theta=0.04, xi=0.01, rho=0.0, v0=0.04)
    semi = heston_call_semi(100, 100, 1, 0.03, p)
    # BSM with sigma = sqrt(0.04) = 0.2 gives ~8.43
    assert 7.5 < semi < 9.5
