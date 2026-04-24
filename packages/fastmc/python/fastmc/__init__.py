"""fastmc — C++ Monte Carlo kernel.

Build:
    pip install ./packages/fastmc

Usage:
    from fastmc import mc_euro_call
    res = mc_euro_call(100, 100, 1.0, 0.03, 0.0, 0.2, n_paths=1_000_000, n_steps=252)
    print(res.price, res.stderr)
"""

from __future__ import annotations

try:
    from fastmc._fastmc import mc_euro_call, MCResult  # type: ignore
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

    class MCResult:  # type: ignore[no-redef]
        """Python fallback matching native signature."""
        def __init__(self, price: float, stderr: float):
            self.price = price
            self.stderr = stderr

        def __repr__(self) -> str:
            return f"<MCResult price={self.price:.6f} stderr={self.stderr:.6f}>"

    def mc_euro_call(S0: float, K: float, T: float, r: float, q: float, sigma: float,
                     n_paths: int, n_steps: int, seed: int = 42,
                     n_threads: int = 0) -> MCResult:  # type: ignore[no-redef]
        """Pure-numpy fallback when C++ extension is not built."""
        import numpy as np

        rng = np.random.default_rng(seed)
        half = n_paths // 2
        dt = T / n_steps
        drift = (r - q - 0.5 * sigma**2) * dt
        diff = sigma * (dt ** 0.5)
        Z = rng.standard_normal(size=(half, n_steps))
        logS1 = (drift + diff * Z).sum(axis=1)
        logS2 = (drift - diff * Z).sum(axis=1)
        import numpy as _np
        S1 = S0 * _np.exp(logS1)
        S2 = S0 * _np.exp(logS2)
        import math
        disc = math.exp(-r * T)
        p = 0.5 * (_np.maximum(S1 - K, 0) + _np.maximum(S2 - K, 0)) * disc
        return MCResult(float(p.mean()), float(p.std(ddof=1) / (half ** 0.5)))


__all__ = ["mc_euro_call", "MCResult", "_HAS_NATIVE"]
