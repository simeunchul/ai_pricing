import numpy as np

from ai_pricing.deep_calib.sampler import sample_heston_params, PARAM_RANGES
from ai_pricing.deep_calib.surface import N_POINTS, iv_surface
from pricing.heston import HestonParams


def test_sampler_ranges():
    X = sample_heston_params(500, seed=0)
    assert X.shape == (500, 5)
    for i, key in enumerate(("kappa", "theta", "xi", "rho", "v0")):
        lo, hi = PARAM_RANGES[key]
        assert X[:, i].min() >= lo - 1e-6
        assert X[:, i].max() <= hi + 1e-6


def test_iv_surface_shape():
    p = HestonParams(kappa=1.5, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
    s = iv_surface(p, S=1.0, r=0.02)
    assert s.shape == (N_POINTS,)
    # IVs should be mostly in reasonable range
    valid = s[~np.isnan(s)]
    assert (valid > 0.01).all() and (valid < 2.0).all()
