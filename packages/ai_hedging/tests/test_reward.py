import numpy as np

from ai_hedging.reward import cvar_loss, max_drawdown, mean_var_loss, sharpe


def test_cvar_positive_on_losses():
    pnl = np.array([-10.0, -5, -1, 0, 1, 2, 3, 4, 5, 6])
    c = cvar_loss(pnl, alpha=0.2)
    assert c > 0


def test_mean_var_zero_on_constant():
    pnl = np.ones(100) * 3.0
    assert abs(mean_var_loss(pnl, lam=1.0) - (-3.0)) < 1e-10


def test_sharpe_positive_on_positive_mean():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.1, 1.0, size=1000)
    assert sharpe(pnl) > 0


def test_max_dd_negative():
    eq = np.array([100, 110, 105, 90, 95, 120])
    dd = max_drawdown(eq)
    assert dd < 0
