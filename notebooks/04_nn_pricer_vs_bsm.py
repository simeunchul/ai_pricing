"""Notebook W4 — NN Pricer vs BSM."""

# %%
# Prerequisite: python -m ai_pricing.nn_pricer.train --n 100000 --epochs 30
import numpy as np
import time
from pathlib import Path

from pricing.bsm import BSMInputs, call_price
from ai_pricing.nn_pricer.data import sample_inputs, label_bsm

MODEL = "models/nn_pricer.pt"
if not Path(MODEL).exists():
    print(f"model not found at {MODEL}. Run:\n  python -m ai_pricing.nn_pricer.train --n 100000 --epochs 30")
else:
    from ai_pricing.nn_pricer.infer import load_pricer, price_batch

    X = sample_inputs(5000, seed=999)
    y = label_bsm(X)

    model, dev = load_pricer(MODEL)
    t0 = time.perf_counter()
    y_hat = price_batch(model, X, device=dev)
    elapsed = time.perf_counter() - t0
    rel = np.abs(y_hat - y) / np.maximum(y, 1e-6)
    print(f"Mean rel err: {rel.mean():.4%}")
    print(f"p95 rel err : {np.quantile(rel, 0.95):.4%}")
    print(f"Total {len(X)} samples in {elapsed*1000:.0f}ms  ({elapsed*1e6/len(X):.1f} µs/sample)")
