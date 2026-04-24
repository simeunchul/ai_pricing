import numpy as np

from experiments.metrics import (
    hedging_stats_from_arrays,
    pricing_stats_from_arrays,
)


def test_pricing_stats_zero_error_on_identity():
    truth = np.array([1.0, 2.0, 3.0])
    pred = truth.copy()
    s = pricing_stats_from_arrays("id", pred, truth, inference_ms=0.5)
    assert s.mean_err == 0.0 and s.mean_rel_err == 0.0


def test_hedging_stats_sensible():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.0, 1.0, size=500)
    s = hedging_stats_from_arrays("test", pnl, turnover=0.5, inference_ms=1.0)
    assert s.cvar_5 > 0
    assert s.std_pnl > 0
    assert s.max_dd <= 0
