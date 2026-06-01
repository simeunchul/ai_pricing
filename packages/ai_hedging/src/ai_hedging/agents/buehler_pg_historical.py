"""Buehler PG-on-CVaR with HISTORICAL bootstrap (B4 GBM → 실시장 path 전환).

v1 (buehler_pg.py): GBM tick (drift + σ √dt × Z) 으로 학습. 합성 환경.
v2 (이 파일):       SPY 일별 historical log-return 에서 block bootstrap.
                    jump, fat tail, vol-clustering 자연 노출.

핵심 변경:
  simulate_batch_historical() — GBM step 을 historical-return sampling 으로 교체

기존 GBM 코드는 보존 (TC=0 sanity 검증용으로 GBM 환경 그대로 필요).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ai_hedging.agents.buehler_pg import (
    BuehlerConfig, HedgingPolicy, _bsm_call_premium, cvar_loss_fn,
)


@dataclass
class HistoricalConfig(BuehlerConfig):
    """Historical bootstrap config — extends BuehlerConfig.

    Adds:
      block_size: block bootstrap 크기 (vol clustering 일부 보존)
      historical_returns: numpy array of log-returns (set externally)
    """
    block_size: int = 5
    historical_returns: np.ndarray = field(default_factory=lambda: np.array([]))


def fetch_spy_returns(start: str = "2014-01-01", end: str = "2026-04-26",
                      cache_path: str = "data/macro_events/spy_returns.csv") -> np.ndarray:
    """SPY 일별 log-return fetch with caching.

    Returns
    -------
    log_returns : np.ndarray (T,)
    """
    from pathlib import Path
    import time

    cache = Path(cache_path)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400:
        print(f"[spy] cache hit: {cache}")
        lines = cache.read_text(encoding="utf-8").splitlines()[1:]
        return np.array([float(line.split(",")[1]) for line in lines])

    import yfinance as yf
    print(f"[spy] fetching SPY {start} → {end} ...")
    df = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=True)
    if df.empty:
        raise RuntimeError("yfinance returned empty for SPY")
    closes = df["Close"].values
    log_rets = np.diff(np.log(closes))

    cache.parent.mkdir(parents=True, exist_ok=True)
    dates = [d.strftime("%Y-%m-%d") for d in df.index.date[1:]]
    cache.write_text(
        "date,log_return\n" + "\n".join(f"{d},{r}" for d, r in zip(dates, log_rets)),
        encoding="utf-8",
    )
    print(f"[spy] {len(log_rets)} daily log-returns saved → {cache}")
    return log_rets


def annualized_vol(log_returns: np.ndarray) -> float:
    """일별 log-return → 연환산 vol (252 trading days)."""
    return float(log_returns.std() * np.sqrt(252))


def simulate_batch_historical(policy: HedgingPolicy, cfg: HistoricalConfig,
                               batch_size: int, device: str = "cpu",
                               seed: int | None = None) -> torch.Tensor:
    """Historical block-bootstrap batch hedging simulation.

    GBM step 대신 historical log-return 에서 block 단위로 sampling.
    """
    rng = np.random.default_rng(seed)
    B = cfg.batch_size if batch_size <= 0 else batch_size
    n_steps = cfg.n_steps
    block = cfg.block_size

    rets = cfg.historical_returns
    if len(rets) == 0:
        raise RuntimeError("historical_returns is empty — call fetch_spy_returns() first")

    n_blocks = (n_steps + block - 1) // block
    starts = rng.integers(0, len(rets) - block, size=(B, n_blocks))
    sampled = np.empty((B, n_blocks * block), dtype=np.float32)
    for j in range(n_blocks):
        for k in range(block):
            sampled[:, j * block + k] = rets[starts[:, j] + k]
    log_ret_tensor = torch.from_numpy(sampled[:, :n_steps]).to(device)

    dt = cfg.T / cfg.n_steps
    exp_r_dt = math.exp(cfg.r * dt)
    premium = _bsm_call_premium(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.q, cfg.sigma)

    S = torch.full((B,), float(cfg.S0), device=device)
    cash = torch.full((B,), float(premium), device=device)
    hedge = torch.zeros(B, device=device)

    for t in range(cfg.n_steps):
        dt_rem = cfg.T * (1 - t / cfg.n_steps)
        log_m = torch.log((S / cfg.K).clamp_min(1e-9))
        t_rem_norm = torch.full((B,), dt_rem / cfg.T, device=device)
        sigma_sqrt_t = torch.full((B,), cfg.sigma * math.sqrt(max(dt_rem, 1e-6)),
                                   device=device)
        cash_norm = cash / max(premium, 1e-6)
        obs = torch.stack([log_m, t_rem_norm, hedge, sigma_sqrt_t, cash_norm], dim=-1)

        new_hedge = policy(obs)
        tc = cfg.tc_rate * torch.abs(new_hedge - hedge) * S
        cash = cash - tc
        hedge = new_hedge

        S_next = S * torch.exp(log_ret_tensor[:, t])
        hedge_pnl = hedge * (S_next - S)
        cash = cash * exp_r_dt + hedge_pnl
        S = S_next

    if cfg.opt == "call":
        payoff = torch.relu(S - cfg.K)
    else:
        payoff = torch.relu(cfg.K - S)
    close_tc = cfg.tc_rate * torch.abs(hedge) * S
    cash = cash - close_tc
    terminal_pnl = cash - payoff * cfg.notional
    return terminal_pnl


def train_buehler_historical(
    cfg: HistoricalConfig,
    out_path: str = "models/buehler_historical.pt",
    device: str = "cpu",
    log_every: int = 10,
    eval_batch: int = 8000,
) -> dict:
    """Train Buehler policy on historical bootstrap paths."""
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
        pnl = simulate_batch_historical(policy, cfg, cfg.batch_size,
                                         device=device, seed=cfg.seed + ep)
        loss = cvar_loss_fn(pnl, cfg.cvar_alpha)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

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

    policy.eval()
    with torch.no_grad():
        eval_pnl = simulate_batch_historical(policy, cfg, eval_batch,
                                              device=device,
                                              seed=cfg.seed + 9999).cpu().numpy()
    final = {
        "mean_pnl": float(eval_pnl.mean()),
        "std_pnl": float(eval_pnl.std()),
        "cvar_5": float(-np.sort(eval_pnl)[:int(0.05 * len(eval_pnl))].mean()),
        "p05": float(np.quantile(eval_pnl, 0.05)),
    }
    print(f"\n[FINAL eval n={eval_batch}]  CVaR={final['cvar_5']:.4f}  "
          f"mean={final['mean_pnl']:+.4f}  std={final['std_pnl']:.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "cfg": cfg.__dict__}, out_path)
    return {"history": history, "final": final, "model_path": out_path,
            "eval_pnl": eval_pnl}


__all__ = [
    "HistoricalConfig", "fetch_spy_returns", "annualized_vol",
    "simulate_batch_historical", "train_buehler_historical",
]
