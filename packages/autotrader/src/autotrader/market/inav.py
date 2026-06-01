"""ETF iNAV estimator from constituent basket prices.

We do not have the issuer's PDF (Portfolio Deposit File), so absolute iNAV in
KRW cannot be reconstructed from index weights alone. Instead we use:

  1) **Calibration (one-shot)** — at the first tick where ALL declared
     constituents have valid prices, fit a single scaling factor so that
     iNAV ≡ ETF price. The leg-set used for calibration is *frozen*.
  2) **Estimation (every tick after)** — recompute the basket using the same
     frozen leg-set; multiply by the locked factor.

If at estimation time any frozen leg is missing a price, the estimator
returns 0.0 (interpreted by callers as "no estimate this tick — do not
trade"). This is critical: if calibration was done with a partial basket,
the locked factor would be inflated/deflated, and once the missing legs
came back online the iNAV would jump 10–25%, generating spurious deviation
signals and bad trades.

For test/demo use without prior calibration, `estimate()` falls back to the
loose `_raw_basket()` (covered legs only, renormalized) × `issuer_factor`.
This pre-calibration mode is *not* used by production runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InavEstimator:
    """Strict iNAV estimator with frozen-leg-set calibration.

    Example:
        est = InavEstimator(constituents={"005930": 0.30, "000660": 0.07, ...})
        # first tick where ALL 10 names have valid prices:
        factor = est.calibrate(etf_price=101_460, prices={...all 10...})
        # subsequent ticks (must include all calibrated legs):
        inav = est.estimate(prices)   # 0.0 if any calibrated leg missing
    """
    constituents: dict[str, float]
    cash_weight: float = 0.0           # informational only (uncovered weight)
    issuer_factor: float = 1.0         # used only in pre-calibration fallback
    _calibration_factor: float | None = field(default=None, repr=False)
    _calibrated_legs: tuple[str, ...] = field(default_factory=tuple, repr=False)

    # ------------------------------------------------------------------ basket

    def _raw_basket(self, prices: dict[str, float]) -> float:
        """Loose basket: covered legs only, renormalized by their weight sum.

        Used (a) by tests that check pre-calibration estimation, and
        (b) by external callers that want a quick approximate basket value.
        Production calibration/estimation must NOT use this — it is the
        source of the partial-coverage drift bug.
        """
        s = 0.0
        covered_w = 0.0
        for sym, w in self.constituents.items():
            p = prices.get(sym)
            if p is None or p <= 0:
                continue
            s += w * p
            covered_w += w
        if covered_w <= 0:
            return 0.0
        return s / covered_w

    def _raw_basket_strict(
        self, prices: dict[str, float], legs: tuple[str, ...]
    ) -> float | None:
        """Strict basket: every leg in `legs` must have a positive price.

        Returns None if any leg is missing or non-positive — that signals the
        caller to skip the tick rather than fall back to a degraded estimate.
        """
        s = 0.0
        cw = 0.0
        for sym in legs:
            p = prices.get(sym)
            if p is None or p <= 0:
                return None
            w = self.constituents.get(sym, 0.0)
            s += w * p
            cw += w
        if cw <= 0:
            return None
        return s / cw

    # --------------------------------------------------------------- calibrate

    def calibrate(self, etf_price: float, prices: dict[str, float]) -> float:
        """Calibrate ONLY when every constituent has a valid price.

        Returns the locked factor on success, or 0.0 if calibration was
        deferred (caller should keep trying on subsequent ticks).
        """
        all_legs = tuple(self.constituents.keys())
        raw = self._raw_basket_strict(prices, all_legs)
        if raw is None or raw <= 0 or etf_price <= 0:
            return 0.0
        self._calibration_factor = etf_price / raw
        self._calibrated_legs = all_legs
        return self._calibration_factor

    @property
    def calibrated(self) -> bool:
        return self._calibration_factor is not None

    @property
    def calibrated_legs(self) -> tuple[str, ...]:
        return self._calibrated_legs

    # ---------------------------------------------------------------- estimate

    def estimate(self, prices: dict[str, float]) -> float:
        """Strict estimate. Returns 0.0 if any frozen leg is missing.

        Pre-calibration fallback: loose basket × issuer_factor (test path).
        """
        if self._calibration_factor is not None and self._calibrated_legs:
            raw = self._raw_basket_strict(prices, self._calibrated_legs)
            if raw is None:
                return 0.0
            return raw * self._calibration_factor
        # Fallback for pre-calibration use (tests / one-shot inav_from_basket).
        raw = self._raw_basket(prices)
        return raw * self.issuer_factor


def inav_from_basket(
    weights: dict[str, float],
    prices: dict[str, float],
    cash_weight: float = 0.0,
) -> float:
    return InavEstimator(constituents=weights, cash_weight=cash_weight).estimate(prices)


def deviation(etf_price: float, inav: float) -> float:
    """Signed percentage deviation. 0 if inav non-positive (i.e. unavailable)."""
    if inav <= 0:
        return 0.0
    return (etf_price - inav) / inav


__all__ = ["InavEstimator", "inav_from_basket", "deviation"]
