"""Fast batch inference for NN Pricer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_pricing.nn_pricer.data import features_from_raw


def load_pricer(path: str | Path, device: str | None = None):
    import torch
    from ai_pricing.nn_pricer.model import NNPricer, NNPricerConfig

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    cfg = NNPricerConfig(**ckpt["cfg"])
    model = NNPricer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device


def price_batch(model, X_raw: np.ndarray, device: str = "cpu") -> np.ndarray:
    """X_raw: (N, 6) = [S, K, T, r, q, sigma]. Returns prices (N,) in raw units."""
    import torch

    feats = features_from_raw(X_raw)
    with torch.no_grad():
        x = torch.from_numpy(feats).to(device)
        out = model(x).cpu().numpy()
    # target was C/K, so multiply back by K
    return out * X_raw[:, 1]


__all__ = ["load_pricer", "price_batch"]
