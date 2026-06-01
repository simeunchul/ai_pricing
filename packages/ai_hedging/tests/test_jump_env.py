"""Merton jump-diffusion 옵션 호환성 + 거동 테스트.

핵심:
  1. jump_intensity=0 (default) 시 path 가 기존 GBM-only 와 bit-exact 동일
     → 기존 학습된 체크포인트가 깨지지 않음
  2. jump_intensity>0 시 PnL 분포의 꼬리가 두꺼워짐 (fat tail 발생)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai_hedging.agents.buehler_pg import (
    BuehlerConfig, HedgingPolicy, simulate_batch,
)

try:
    from ai_hedging.env import HedgingEnv, HedgingEnvConfig
    _HAS_GYM_ENV = True
except ImportError:
    _HAS_GYM_ENV = False


# --------------------------------------------------------------------------- #
# 1. simulate_batch (PyTorch 버전) — λ=0 시 기존 결과 100% 동일
# --------------------------------------------------------------------------- #
def test_simulate_batch_jump_off_equals_legacy():
    cfg_legacy = BuehlerConfig(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0, sigma=0.20,
        n_steps=30, tc_rate=0.003, batch_size=512, seed=0,
    )
    cfg_jump_off = BuehlerConfig(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0, sigma=0.20,
        n_steps=30, tc_rate=0.003, batch_size=512, seed=0,
        jump_intensity=0.0, jump_mean=-0.05, jump_std=0.10,
    )
    torch.manual_seed(0)
    pol = HedgingPolicy(action_low=0.0, action_high=1.0)
    pnl_a = simulate_batch(pol, cfg_legacy, batch_size=512, device="cpu", seed=42)
    torch.manual_seed(0)
    pol2 = HedgingPolicy(action_low=0.0, action_high=1.0)
    pol2.load_state_dict(pol.state_dict())
    pnl_b = simulate_batch(pol2, cfg_jump_off, batch_size=512, device="cpu", seed=42)
    assert torch.allclose(pnl_a, pnl_b, atol=1e-12, rtol=0), (
        "λ=0 path must be bit-exact identical to legacy GBM"
    )


def test_simulate_batch_jump_on_creates_fat_tail():
    """λ>0 일 때 PnL 분포 꼬리가 두꺼워져야 함."""
    common = dict(
        S0=100.0, K=100.0, T=30/365, r=0.02, q=0.0, sigma=0.20,
        n_steps=30, tc_rate=0.003, batch_size=4096, seed=0,
    )
    cfg_no_jump  = BuehlerConfig(**common)
    cfg_w_jump   = BuehlerConfig(**common,
        jump_intensity=20.0,    # 1년에 20번 (큰 값으로 효과 보장)
        jump_mean=-0.02,        # 평균적으로 음의 jump
        jump_std=0.05,
    )
    torch.manual_seed(0)
    pol = HedgingPolicy(action_low=0.0, action_high=1.0)
    pnl_no   = simulate_batch(pol, cfg_no_jump, 4096, "cpu", seed=99).detach().numpy()
    torch.manual_seed(0)
    pol2 = HedgingPolicy(action_low=0.0, action_high=1.0)
    pol2.load_state_dict(pol.state_dict())
    pnl_jump = simulate_batch(pol2, cfg_w_jump, 4096, "cpu", seed=99).detach().numpy()

    # CVaR@5% (worst 5% mean loss) 가 jump 환경에서 더 큼 (= 꼬리 손실 ↑)
    n_tail = int(0.05 * len(pnl_no))
    cvar_no   = -np.sort(pnl_no)[:n_tail].mean()
    cvar_jump = -np.sort(pnl_jump)[:n_tail].mean()
    assert cvar_jump > cvar_no, (
        f"jump should create fat tail. cvar(no)={cvar_no:.4f} cvar(jump)={cvar_jump:.4f}"
    )


# --------------------------------------------------------------------------- #
# 2. gym HedgingEnv — λ=0 시 기존과 동일 path
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_GYM_ENV, reason="gymnasium not installed")
def test_gym_env_jump_off_equals_legacy():
    cfg_legacy = HedgingEnvConfig(seed=123, reward_shaping=False)
    cfg_jump_off = HedgingEnvConfig(
        seed=123, reward_shaping=False,
        jump_intensity=0.0, jump_mean=-0.05, jump_std=0.10,
    )
    env_a = HedgingEnv(cfg_legacy)
    env_b = HedgingEnv(cfg_jump_off)
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    for _ in range(10):
        a = np.array([0.5], dtype=np.float32)
        oa, ra, da, _, _ = env_a.step(a)
        ob, rb, db, _, _ = env_b.step(a)
        np.testing.assert_allclose(oa, ob, atol=1e-15)
        assert abs(ra - rb) < 1e-12
        assert da == db


@pytest.mark.skipif(not _HAS_GYM_ENV, reason="gymnasium not installed")
def test_gym_env_jump_on_changes_path():
    """λ>0 시 같은 seed 라도 (jump RNG 추가 호출 때문에) path 통계가 달라져야 함."""
    cfg_no = HedgingEnvConfig(seed=42, reward_shaping=False)
    cfg_jp = HedgingEnvConfig(
        seed=42, reward_shaping=False,
        jump_intensity=30.0, jump_mean=-0.03, jump_std=0.08,
    )
    finals_no, finals_jp = [], []
    for p in range(200):
        env = HedgingEnv(cfg_no)
        env.reset(seed=p)
        for _ in range(cfg_no.n_steps):
            obs, r, done, _, info = env.step(np.array([0.5], dtype=np.float32))
            if done:
                finals_no.append(info["terminal_pnl"])
                break
        env = HedgingEnv(cfg_jp)
        env.reset(seed=p)
        for _ in range(cfg_jp.n_steps):
            obs, r, done, _, info = env.step(np.array([0.5], dtype=np.float32))
            if done:
                finals_jp.append(info["terminal_pnl"])
                break
    arr_no, arr_jp = np.array(finals_no), np.array(finals_jp)
    # jump 환경에서 std 가 (대체로) 더 크다
    assert arr_jp.std() > arr_no.std(), (
        f"jump path std should be larger. std(no)={arr_no.std():.4f} std(jp)={arr_jp.std():.4f}"
    )
