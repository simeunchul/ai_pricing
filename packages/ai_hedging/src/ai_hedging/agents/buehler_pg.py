"""Buehler 2019 직접 구현 — Policy gradient on CVaR loss.

원논문 (Buehler-Gonon-Teichmann-Wood, "Deep Hedging" QF 2019, eq.13-14):
  Loss = CVaR_α [ payoff − Σ_t π_θ(s_t) · ΔS_t + Σ_t TC · |Δπ_θ| ]

PPO 와 다른 점:
  - gym env 안 씀. PyTorch tensor 로 batch GBM 시뮬 (differentiable).
  - reward shaping 없음. terminal CVaR loss 만 backward.
  - policy → action → S_path → terminal_pnl → CVaR 까지 gradient 직통.
  - BSM attractor 없음 (loss 자체가 CVaR 라 BSM 으로 끌릴 이유 없음).

Why this works (vs 5번 실패한 PPO 시도들):
  PPO 는 E[reward] 최적화. CVaR 는 batch-level operator 라 PPO advantage
  공식에 들어가지 못함. Buehler 의 답: CVaR loss 자체를 batch loss 로
  만들고 그 gradient 로 정책 업데이트.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class BuehlerConfig:
    """Hedging environment + training config."""
    S0: float = 100.0
    K: float = 100.0
    T: float = 30 / 365
    r: float = 0.02
    q: float = 0.0
    sigma: float = 0.20
    n_steps: int = 30
    tc_rate: float = 0.003
    notional: float = 1.0
    opt: str = "call"

    # Policy
    hidden: tuple[int, ...] = (64, 64, 32)
    activation: str = "relu"

    # Training
    batch_size: int = 4096
    lr: float = 1e-3
    n_epochs: int = 200
    cvar_alpha: float = 0.05      # tail level for CVaR_α
    grad_clip: float = 1.0
    seed: int = 0

    # Action bound
    action_low: float = 0.0
    action_high: float = 1.0

    # Merton jump-diffusion (optional). Defaults 0 → pure GBM, identical to legacy.
    # Activated when jump_intensity > 0. Policy 입력에는 jump 신호가 안 들어가므로
    # Buehler "예측 안 함" 철학은 보존된다 (환경만 비정형화).
    jump_intensity: float = 0.0       # λ — Poisson jumps per year
    jump_mean: float = 0.0            # μJ — log-normal jump mean (log scale)
    jump_std: float = 0.0             # σJ — log-normal jump std
    jump_compensator: bool = True     # subtract λκ dt from drift


def _bsm_call_premium(S0, K, T, r, q, sigma):
    """BSM call closed-form (used for initial cash = sold premium)."""
    from scipy.stats import norm
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


class HedgingPolicy(nn.Module):
    """π_θ(state) → hedge ∈ [low, high].

    State (5-dim, same as gym env so policy is comparable):
      [log(S/K), T_remaining/T_total, current_hedge, σ√T_rem, cash/initial_premium]
    """

    def __init__(self, hidden=(64, 64, 32), activation="relu",
                 action_low=0.0, action_high=1.0):
        super().__init__()
        self.action_low = action_low
        self.action_range = action_high - action_low

        Act = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        layers = []
        in_dim = 5
        for h in hidden:
            layers += [nn.Linear(in_dim, h), Act()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        # obs: (B, 5). Output sigmoid → [0, 1] then scale to [low, high].
        raw = self.net(obs).squeeze(-1)
        return self.action_low + self.action_range * torch.sigmoid(raw)


def simulate_batch(policy: HedgingPolicy, cfg: BuehlerConfig,
                   batch_size: int, device: str = "cpu",
                   seed: int | None = None) -> torch.Tensor:
    """Differentiable batch GBM hedging simulation.

    Returns
    -------
    terminal_pnl : Tensor (batch_size,)
        gradient flows back through policy.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    B = batch_size
    dt = cfg.T / cfg.n_steps
    drift_dt = (cfg.r - cfg.q - 0.5 * cfg.sigma ** 2) * dt
    # Merton jump compensator (only when jump active; keeps risk-neutral drift)
    if cfg.jump_intensity > 0.0 and cfg.jump_compensator:
        kappa = math.exp(cfg.jump_mean + 0.5 * cfg.jump_std ** 2) - 1.0
        drift_dt -= cfg.jump_intensity * kappa * dt
    diff_dt = cfg.sigma * math.sqrt(dt)
    exp_r_dt = math.exp(cfg.r * dt)
    lam_dt = cfg.jump_intensity * dt  # Poisson rate per step (0 if disabled)

    # initial premium (sold short call, fixed scalar — same for all paths)
    premium = _bsm_call_premium(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma)

    # state tensors
    S = torch.full((B,), float(cfg.S0), device=device)
    cash = torch.full((B,), float(premium), device=device)
    hedge = torch.zeros(B, device=device)

    for t in range(cfg.n_steps):
        # observation (5-dim, matches gym env's _obs)
        dt_rem = cfg.T * (1 - t / cfg.n_steps)
        log_m = torch.log((S / cfg.K).clamp_min(1e-9))
        t_rem_norm = torch.full((B,), dt_rem / cfg.T, device=device)
        sigma_sqrt_t = torch.full((B,), cfg.sigma * math.sqrt(max(dt_rem, 1e-6)),
                                  device=device)
        cash_norm = cash / max(premium, 1e-6)
        obs = torch.stack([log_m, t_rem_norm, hedge, sigma_sqrt_t, cash_norm], dim=-1)

        # policy: new hedge
        new_hedge = policy(obs)

        # transaction cost (smooth, differentiable)
        tc = cfg.tc_rate * torch.abs(new_hedge - hedge) * S
        cash = cash - tc
        hedge = new_hedge

        # GBM step (+ optional Merton jump)
        Z = torch.randn(B, device=device, generator=gen)
        if lam_dt > 0.0:
            # Per-path Poisson(λdt) jump count, then sum of n iid Normal(μJ,σJ²)
            # = Normal(n·μJ, n·σJ²). Vectorized, gradient-free path noise.
            n_jumps = torch.poisson(
                torch.full((B,), lam_dt, device=device), generator=gen
            )
            n_float = n_jumps
            jump_Z = torch.randn(B, device=device, generator=gen)
            log_jump = n_float * cfg.jump_mean + torch.sqrt(n_float) * cfg.jump_std * jump_Z
            S_next = S * torch.exp(drift_dt + diff_dt * Z + log_jump)
        else:
            S_next = S * torch.exp(drift_dt + diff_dt * Z)

        # cash + hedge pnl
        hedge_pnl = hedge * (S_next - S)
        cash = cash * exp_r_dt + hedge_pnl
        S = S_next

    # terminal payoff & close hedge cost
    if cfg.opt == "call":
        payoff = torch.relu(S - cfg.K)
    else:
        payoff = torch.relu(cfg.K - S)
    close_tc = cfg.tc_rate * torch.abs(hedge) * S
    cash = cash - close_tc
    terminal_pnl = cash - payoff * cfg.notional
    return terminal_pnl


def cvar_loss_fn(pnl: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """CVaR_α (one-sided lower tail loss).

    CVaR = -E[PnL | PnL ≤ q_α]  (positive when expected tail loss > 0)

    pnl 이 손익(positive=이익) 일 때 CVaR 가 양수면 tail 에 손실 있음.
    이 loss 최소화 = tail 평균 손실 줄이기.
    """
    n = pnl.numel()
    n_tail = max(int(alpha * n), 1)
    sorted_pnl, _ = torch.sort(pnl)
    tail = sorted_pnl[:n_tail]   # bottom alpha fraction
    return -tail.mean()


def mean_var_loss_fn(pnl: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return -pnl.mean() + lam * pnl.var()


def train_buehler(
    cfg: BuehlerConfig | None = None,
    loss_type: str = "cvar",          # "cvar" | "mean_var" | "exp_utility"
    out_path: str = "models/buehler_tc03.pt",
    device: str = "cpu",
    log_every: int = 10,
    eval_batch: int = 8000,
) -> dict:
    """Train policy using direct CVaR loss gradient."""
    cfg = cfg or BuehlerConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    policy = HedgingPolicy(
        hidden=cfg.hidden, activation=cfg.activation,
        action_low=cfg.action_low, action_high=cfg.action_high,
    ).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)

    history = {"epoch": [], "cvar": [], "mean_pnl": [], "std_pnl": []}

    for ep in range(cfg.n_epochs):
        policy.train()
        pnl = simulate_batch(policy, cfg, cfg.batch_size, device=device, seed=cfg.seed + ep)

        if loss_type == "cvar":
            loss = cvar_loss_fn(pnl, cfg.cvar_alpha)
        elif loss_type == "mean_var":
            loss = mean_var_loss_fn(pnl, lam=1.0)
        else:
            raise ValueError(loss_type)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        # diagnostic
        with torch.no_grad():
            mean_pnl = pnl.mean().item()
            std_pnl = pnl.std().item()
            cvar = cvar_loss_fn(pnl, cfg.cvar_alpha).item()
        history["epoch"].append(ep)
        history["cvar"].append(cvar)
        history["mean_pnl"].append(mean_pnl)
        history["std_pnl"].append(std_pnl)
        if ep % log_every == 0 or ep == cfg.n_epochs - 1:
            print(f"[ep {ep:3d}/{cfg.n_epochs}]  loss={loss.item():+.4f}  "
                  f"CVaR={cvar:.4f}  mean_pnl={mean_pnl:+.4f}  std={std_pnl:.4f}",
                  flush=True)

    # Eval on a fresh batch
    policy.eval()
    with torch.no_grad():
        eval_pnl = simulate_batch(policy, cfg, eval_batch, device=device,
                                   seed=cfg.seed + 9999).cpu().numpy()
    final = {
        "mean_pnl": float(eval_pnl.mean()),
        "std_pnl": float(eval_pnl.std()),
        "cvar_5": float(-np.sort(eval_pnl)[:int(0.05 * len(eval_pnl))].mean()),
        "p05": float(np.quantile(eval_pnl, 0.05)),
        "max_dd": float(np.minimum.accumulate(eval_pnl - eval_pnl.mean()).min()),
    }
    print(f"\n[FINAL eval n={eval_batch}]  CVaR={final['cvar_5']:.4f}  "
          f"mean={final['mean_pnl']:+.4f}  std={final['std_pnl']:.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "cfg": cfg.__dict__}, out_path)
    return {"history": history, "final": final, "model_path": out_path,
            "eval_pnl": eval_pnl}


def evaluate(model_path: str, cfg: BuehlerConfig, n_paths: int = 8000,
             device: str = "cpu", seed: int = 9999) -> dict:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    policy = HedgingPolicy(
        hidden=cfg.hidden, activation=cfg.activation,
        action_low=cfg.action_low, action_high=cfg.action_high,
    ).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    with torch.no_grad():
        pnl = simulate_batch(policy, cfg, n_paths, device=device, seed=seed).cpu().numpy()
    return {
        "pnl_array": pnl,
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "cvar_5": float(-np.sort(pnl)[:int(0.05 * len(pnl))].mean()),
        "p05": float(np.quantile(pnl, 0.05)),
    }


__all__ = [
    "BuehlerConfig", "HedgingPolicy", "simulate_batch",
    "cvar_loss_fn", "mean_var_loss_fn",
    "train_buehler", "evaluate",
]
