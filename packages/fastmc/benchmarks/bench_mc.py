"""Benchmark: numpy vs (optional) numba vs C++ OpenMP.

Target metrics (Week 9 plan):
    numpy     ~ 12 s  (1x)
    numba     ~  4 s  (3x)
    cpp-omp   ~0.45 s (~27x)
"""

from __future__ import annotations

import math
import time

import numpy as np


def numpy_mc(S0=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2,
             n_paths=1_000_000, n_steps=252, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    half = n_paths // 2
    Z = rng.standard_normal((half, n_steps))
    logS1 = (drift + diff * Z).sum(axis=1)
    logS2 = (drift - diff * Z).sum(axis=1)
    S1 = S0 * np.exp(logS1)
    S2 = S0 * np.exp(logS2)
    disc = math.exp(-r * T)
    p = 0.5 * (np.maximum(S1 - K, 0) + np.maximum(S2 - K, 0)) * disc
    return float(p.mean())


def numba_mc(S0=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2,
             n_paths=1_000_000, n_steps=252, seed=42):
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(parallel=True, fastmath=True)
    def _inner(S0, K, T, r, q, sigma, n_paths, n_steps):
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diff = sigma * math.sqrt(dt)
        total = 0.0
        half = n_paths // 2
        for i in numba.prange(half):
            logS1 = 0.0
            logS2 = 0.0
            np.random.seed(42 + i)  # numba rand
            for _ in range(n_steps):
                z = np.random.normal()
                logS1 += drift + diff * z
                logS2 += drift - diff * z
            s1 = S0 * math.exp(logS1)
            s2 = S0 * math.exp(logS2)
            total += 0.5 * (max(s1 - K, 0.0) + max(s2 - K, 0.0))
        return total / half * math.exp(-r * T)

    return _inner(S0, K, T, r, q, sigma, n_paths, n_steps)


def cpp_mc(**kw):
    try:
        from fastmc import mc_euro_call
    except ImportError:
        return None
    res = mc_euro_call(**kw)
    return res.price


def _time(fn, *args, **kw) -> tuple[float, float]:
    t0 = time.perf_counter()
    val = fn(*args, **kw)
    return val, time.perf_counter() - t0


def main():
    common = dict(S0=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2,
                  n_paths=1_000_000, n_steps=252, seed=42)

    p_np, t_np = _time(numpy_mc, **common)
    print(f"numpy       : {p_np:.6f} in {t_np:.2f}s")

    p_nb, t_nb = _time(numba_mc, **common)
    if p_nb is not None:
        print(f"numba       : {p_nb:.6f} in {t_nb:.2f}s")

    p_cpp, t_cpp = _time(cpp_mc, **common)
    if p_cpp is not None:
        print(f"fastmc(C++) : {p_cpp:.6f} in {t_cpp:.2f}s  ({t_np / t_cpp:.1f}x vs numpy)")


if __name__ == "__main__":
    main()
