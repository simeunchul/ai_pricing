import numpy as np

from ai_pricing.nn_pricer.data import (
    SAMPLING_RANGES,
    features_from_raw,
    label_bsm,
    sample_inputs,
)


def test_sample_inputs_within_ranges():
    X = sample_inputs(1000, seed=0)
    assert X.shape == (1000, 6)
    S, K, T, r, q, sigma = X.T
    assert (S / K).min() >= SAMPLING_RANGES["moneyness"][0] - 1e-6
    assert (S / K).max() <= SAMPLING_RANGES["moneyness"][1] + 1e-6
    assert T.min() >= SAMPLING_RANGES["T"][0] - 1e-6
    assert sigma.max() <= SAMPLING_RANGES["sigma"][1] + 1e-6


def test_labels_positive_and_bounded():
    X = sample_inputs(200, seed=1)
    y = label_bsm(X)
    assert (y >= 0).all()
    # call price <= S
    assert (y <= X[:, 0] + 1e-6).all()


def test_features_shape():
    X = sample_inputs(50, seed=2)
    feats = features_from_raw(X)
    assert feats.shape == (50, 5)
    assert feats.dtype == np.float32
