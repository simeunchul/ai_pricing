"""Notebook W3 — 한화투자증권 ELS 재가격.

Scenario: KOSPI200 + HSCEI 2자산 step-down, KI=50%, coupon=3%/6mo, 3Y.
"""

# %%
import numpy as np
from pricing.els.step_down import StepDownELS, price_els

# 가상 스펙 (실제 공시는 data/els_samples/ 에 PDF 로 참고)
prod = StepDownELS(
    S0=np.array([100.0, 100.0]),
    barriers=[0.90, 0.90, 0.85, 0.85, 0.80, 0.75],
    ki_barrier=0.50,
    coupon_rate=0.03,
    maturity_years=3.0,
    obs_per_year=2,
    notional=10_000.0,
)

# %% Base case
res = price_els(prod, r=0.03,
                q=np.array([0.0, 0.0]),
                sigma=np.array([0.22, 0.28]),
                corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
                n_paths=20_000, n_steps_per_year=64, seed=0)
print(f"Base price: {res.price:.0f} (± {res.stderr:.1f})")
print(f"KI hit prob: {res.ki_hit_prob:.2%}")
print(f"Expected life: {res.expected_life:.2f}y")
print(f"Autocall probs per period: {[f'{x:.1%}' for x in res.autocall_prob]}")

# %% ±3% volatility sensitivity
for bump in [-0.03, 0.0, +0.03]:
    vol = np.array([0.22 + bump, 0.28 + bump])
    r = price_els(prod, r=0.03, q=np.zeros(2), sigma=vol,
                  corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
                  n_paths=10_000, n_steps_per_year=64, seed=1)
    print(f"vol bump {bump:+.2f}: price={r.price:.0f}, KI={r.ki_hit_prob:.2%}")
