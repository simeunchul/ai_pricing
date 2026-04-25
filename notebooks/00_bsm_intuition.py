"""BSM 직관 시각화 6패널 — 페이오프 / 시간가치 / GBM / d1·d2 / Greeks / 변동성 효과."""

# %%
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import norm

from pricing.bsm import BSMInputs, call_price, put_price
from pricing.greeks.analytic import call_greeks

rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "docs" / "bsm_intuition.png"

S0, K, r, q, sigma = 100.0, 100.0, 0.03, 0.0, 0.20

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    "Black-Scholes-Merton 직관 (S0=100, K=100, r=3%, sigma=20%)",
    fontsize=15,
    fontweight="bold",
)

# ── (1) 만기 페이오프 ────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
S_grid = np.linspace(60, 140, 200)
call_payoff = np.maximum(S_grid - K, 0.0)
put_payoff = np.maximum(K - S_grid, 0.0)
ax1.plot(S_grid, call_payoff, "b-", lw=2, label="Call payoff = max(S - K, 0)")
ax1.plot(S_grid, put_payoff, "r-", lw=2, label="Put payoff = max(K - S, 0)")
ax1.axvline(K, color="gray", ls="--", alpha=0.5, label=f"K={K:.0f}")
ax1.set_xlabel("만기 주가 S_T")
ax1.set_ylabel("Payoff")
ax1.set_title("① 만기 페이오프 — 옵션의 본질")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# ── (2) 옵션 가치 vs 주가 — 만기까지 시간별 ───────────────────
ax2 = fig.add_subplot(2, 3, 2)
S_grid = np.linspace(60, 140, 100)
for T, color in [(2.0, "C0"), (1.0, "C1"), (0.25, "C2"), (0.01, "C3")]:
    prices = [call_price(BSMInputs(S, K, T, r, q, sigma)) for S in S_grid]
    ax2.plot(S_grid, prices, color=color, lw=2, label=f"T={T:.2f} yr")
ax2.plot(S_grid, np.maximum(S_grid - K, 0), "k--", lw=1, alpha=0.6, label="만기 (T=0)")
ax2.axvline(K, color="gray", ls=":", alpha=0.5)
ax2.set_xlabel("현재 주가 S")
ax2.set_ylabel("Call 가격")
ax2.set_title("② 시간가치 — T가 클수록 곡선이 높아짐")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(alpha=0.3)

# ── (3) GBM 샘플 경로 ───────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
np.random.seed(42)
T_sim, n_steps, n_paths = 1.0, 252, 30
dt = T_sim / n_steps
t_grid = np.linspace(0, T_sim, n_steps + 1)
Z = np.random.randn(n_paths, n_steps)
log_returns = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
log_S = np.cumsum(log_returns, axis=1)
S_paths = S0 * np.exp(np.column_stack([np.zeros(n_paths), log_S]))
for path in S_paths:
    ax3.plot(t_grid, path, alpha=0.5, lw=0.8)
ax3.axhline(K, color="red", ls="--", lw=1.5, label=f"K={K:.0f}")
ax3.axhline(S0, color="black", ls=":", lw=1, alpha=0.5)
ax3.set_xlabel("시간 (year)")
ax3.set_ylabel("주가 S_t")
ax3.set_title("③ GBM 30개 경로 — dS = r*S*dt + sigma*S*dW")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# ── (4) d1, d2와 정규분포 — N(d1), N(d2)의 의미 ─────────────────
ax4 = fig.add_subplot(2, 3, 4)
T_d, sig_d = 1.0, sigma
d1 = (math.log(S0 / K) + (r + 0.5 * sig_d**2) * T_d) / (sig_d * math.sqrt(T_d))
d2 = d1 - sig_d * math.sqrt(T_d)
x = np.linspace(-4, 4, 400)
pdf = norm.pdf(x)
ax4.plot(x, pdf, "k-", lw=1.5)
ax4.fill_between(x, 0, pdf, where=(x <= d1), alpha=0.25, color="blue", label=f"N(d1)={norm.cdf(d1):.3f}")
ax4.fill_between(x, 0, pdf, where=(x <= d2), alpha=0.4, color="red", label=f"N(d2)={norm.cdf(d2):.3f}")
ax4.axvline(d1, color="blue", ls="--", lw=1.2)
ax4.axvline(d2, color="red", ls="--", lw=1.2)
ax4.text(d1 + 0.05, 0.38, f"d1={d1:.3f}", color="blue", fontsize=9)
ax4.text(d2 - 0.95, 0.34, f"d2={d2:.3f}", color="red", fontsize=9)
ax4.set_xlabel("z")
ax4.set_ylabel("표준정규 PDF")
ax4.set_title("④ N(d1), N(d2) — 공식 속 누적확률")
ax4.legend(fontsize=9, loc="upper left")
ax4.grid(alpha=0.3)

# ── (5) Greeks ──────────────────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
S_grid = np.linspace(60, 140, 100)
greeks_T = 0.5
deltas, gammas, vegas, thetas = [], [], [], []
for S in S_grid:
    g = call_greeks(BSMInputs(S, K, greeks_T, r, q, sigma))
    deltas.append(g.delta)
    gammas.append(g.gamma * 50)
    vegas.append(g.vega / 100 * 5)
    thetas.append(g.theta / 365 * 50)
ax5.plot(S_grid, deltas, lw=2, label="Delta (dC/dS)")
ax5.plot(S_grid, gammas, lw=2, label="Gamma ×50")
ax5.plot(S_grid, vegas, lw=2, label="Vega/100 ×5")
ax5.plot(S_grid, thetas, lw=2, label="Theta/일 ×50")
ax5.axvline(K, color="gray", ls=":", alpha=0.5)
ax5.axhline(0, color="black", lw=0.5)
ax5.set_xlabel("현재 주가 S")
ax5.set_ylabel("민감도 (스케일 조정)")
ax5.set_title(f"⑤ Greeks — 가격이 무엇에 얼마나 민감한가 (T={greeks_T})")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# ── (6) 변동성이 가격에 미치는 영향 ────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
sigmas = np.linspace(0.05, 0.80, 100)
for K_val, color, label in [
    (90, "C0", "ITM (K=90)"),
    (100, "C1", "ATM (K=100)"),
    (110, "C2", "OTM (K=110)"),
]:
    prices = [call_price(BSMInputs(S0, K_val, 1.0, r, q, s)) for s in sigmas]
    ax6.plot(sigmas * 100, prices, color=color, lw=2, label=label)
ax6.set_xlabel("변동성 sigma (%)")
ax6.set_ylabel("Call 가격 (T=1yr)")
ax6.set_title("⑥ 변동성이 곧 옵션의 가치 (vol = price)")
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

plt.tight_layout(rect=(0, 0, 1, 0.96))
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved -> {OUT}")
