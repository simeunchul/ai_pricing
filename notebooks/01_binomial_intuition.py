"""Notebook W1 — Binomial Tree intuition.

Save as .ipynb with `jupytext --to notebook notebooks/01_binomial_intuition.py`.
"""

# %% [markdown]
# # Week 1: Binomial Tree Intuition
#
# Risk-neutral pricing with a 1-step tree, then convergence to BSM as N → ∞.

# %%
import numpy as np
import matplotlib.pyplot as plt

from pricing.bsm import BSMInputs, call_price
from pricing.binomial import binomial_european

# %%
inputs = BSMInputs(S=100, K=100, T=1, r=0.03, q=0, sigma=0.2)
bsm = call_price(inputs)
Ns = [5, 10, 50, 100, 500, 1000, 2000]
prices = [binomial_european(inputs, N=N, opt="call") for N in Ns]

# %%
plt.figure(figsize=(8, 4))
plt.semilogx(Ns, prices, "o-", label="Binomial")
plt.axhline(bsm, color="r", linestyle="--", label=f"BSM = {bsm:.4f}")
plt.xlabel("N (tree steps)")
plt.ylabel("Call price")
plt.title("Binomial → BSM convergence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("binomial_convergence.png")
print(f"BSM = {bsm:.6f}, Binomial(N=2000) = {prices[-1]:.6f}, diff = {abs(prices[-1]-bsm):.2e}")
