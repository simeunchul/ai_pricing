"""Training data generation for NN Pricer.

Generate (S, K, T, r, q, sigma) samples across a realistic domain and label
with BSM closed-form price (ground truth for European call).
"""

from __future__ import annotations

import numpy as np

from pricing.bsm import BSMInputs, call_price

# Realistic domain. Outside these ranges the NN will extrapolate poorly (intentional).
SAMPLING_RANGES: dict[str, tuple[float, float]] = {
    "moneyness": (0.6, 1.4),       # S/K
    "T":         (1 / 52, 2.0),    # 1 week → 2 years
    "r":         (0.00, 0.08),
    "q":         (0.00, 0.05),
    "sigma":     (0.05, 0.80),
}


def sample_inputs(n: int, seed: int | None = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.uniform(*SAMPLING_RANGES["moneyness"], size=n)
    T = rng.uniform(*SAMPLING_RANGES["T"], size=n)
    r = rng.uniform(*SAMPLING_RANGES["r"], size=n)
    q = rng.uniform(*SAMPLING_RANGES["q"], size=n)
    sigma = rng.uniform(*SAMPLING_RANGES["sigma"], size=n)
    # Fix K=1.0 and set S = m; this is WLOG for BSM (homogeneous of degree 1 in S,K).
    S = m
    K = np.ones_like(m)
    return np.stack([S, K, T, r, q, sigma], axis=1)


def label_bsm(X: np.ndarray) -> np.ndarray:
    """Label each row with BSM call price."""
    out = np.empty(len(X))
    for i in range(len(X)):
        S, K, T, r, q, sigma = X[i]
        out[i] = call_price(BSMInputs(S, K, T, r, q, sigma))
    return out


def features_from_raw(X: np.ndarray) -> np.ndarray:
    """Turn (S,K,T,r,q,sigma) into NN features [logM, T, r, q, sigma]."""
    S, K, T, r, q, sigma = X.T
    logM = np.log(S / K)
    return np.stack([logM, T, r, q, sigma], axis=1).astype(np.float32)


def generate_training_set(
    n: int = 500_000,
    seed: int | None = 42,
    split: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """Returns dict with train/val/test feature & label tensors (as np arrays)."""
    X = sample_inputs(n, seed=seed)
    y = label_bsm(X)
    feats = features_from_raw(X)
    # target = C / K (K=1 here so target == C; keep the convention for general case)
    targets = (y / X[:, 1]).astype(np.float32)

    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    return {
        "X_train": feats[:n_train],
        "y_train": targets[:n_train],
        "X_val":   feats[n_train:n_train + n_val],
        "y_val":   targets[n_train:n_train + n_val],
        "X_test":  feats[n_train + n_val:],
        "y_test":  targets[n_train + n_val:],
        "raw_test": X[n_train + n_val:],
    }


__all__ = [
    "SAMPLING_RANGES",
    "sample_inputs",
    "label_bsm",
    "features_from_raw",
    "generate_training_set",
]
