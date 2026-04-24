"""Notebook W2 — BSM + Greeks surfaces + IV smile."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from pricing.bsm import BSMInputs, call_price
from pricing.greeks.analytic import call_greeks
from pricing.iv import implied_vol

# %% Greeks surface
Ks = np.linspace(80, 120, 41)
Ts = np.linspace(0.1, 2.0, 20)
delta = np.zeros((len(Ts), len(Ks)))
gamma = np.zeros((len(Ts), len(Ks)))
vega  = np.zeros((len(Ts), len(Ks)))

for i, T in enumerate(Ts):
    for j, K in enumerate(Ks):
        g = call_greeks(BSMInputs(100, K, T, 0.03, 0, 0.2))
        delta[i, j] = g.delta
        gamma[i, j] = g.gamma
        vega[i, j]  = g.vega

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, Z, name in zip(axes, [delta, gamma, vega], ["Delta", "Gamma", "Vega"]):
    im = ax.contourf(Ks, Ts, Z, levels=20, cmap="viridis")
    ax.set_title(name); ax.set_xlabel("K"); ax.set_ylabel("T")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("greeks_surface.png")
