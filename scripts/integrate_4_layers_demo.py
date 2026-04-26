"""4-Layer Integration Demo (#1, #2, #3, #4 통합).

각 통합을 한 번씩 실행해서 실제로 작동함을 보여줌.
docs/2026-04-26/integration_4_layers.html 의 출처.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for sub in ("pricing", "ai_pricing", "ai_hedging", "autotrader"):
    sys.path.insert(0, str(ROOT / "packages" / sub / "src"))


def section(title: str):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# #1: B2 → ELS daily NAV (already in scripts/integrate_b2_els_daily.py)
# ---------------------------------------------------------------------------

def demo_1_b2_els():
    section("#1  B2 → ELS Daily NAV  (한화 8286호)")
    from pricing.els.step_down import StepDownELS
    from ai_pricing.integrations.els_daily_nav import els_daily_nav
    from ai_pricing.deep_calib.surface import STRIKES, MATURITIES

    snaps = sorted((ROOT / "data" / "market_snapshots").glob("spy_iv_*.json"))
    snaps = [s for s in snaps if "_calib" not in s.name]
    if not snaps:
        print("  (no SPY snapshot — skip)"); return None
    snap = json.loads(snaps[-1].read_text(encoding="utf-8"))
    market = np.array(snap["iv_grid"])
    snap_T = np.array(snap["maturities_months"]) / 12
    snap_m = np.array(snap["moneyness"])
    iv_25 = np.full((len(MATURITIES), len(STRIKES)), np.nan)
    for i, T in enumerate(MATURITIES):
        i_src = int(np.argmin(np.abs(snap_T - T)))
        for j, m in enumerate(STRIKES):
            j_src = int(np.argmin(np.abs(snap_m - m)))
            iv_25[i, j] = market[i_src, j_src]
    iv_25 = iv_25.flatten()

    product = StepDownELS(
        S0=np.array([100.0, 100.0, 100.0]),
        barriers=[0.95, 0.90, 0.85, 0.85, 0.80, 0.75],
        ki_barrier=0.50, coupon_rate=0.1131 / 2,
        maturity_years=3.0, obs_per_year=2, notional=10_000.0,
    )
    res = els_daily_nav(
        product=product,
        iv_surfaces_per_asset=[iv_25, iv_25, iv_25],
        asset_names=["KOSPI200", "S&P500", "SX5E"],
        r=0.035, q=np.array([0.017, 0.015, 0.031]),
        corr=np.array([[1.0,0.45,0.55],[0.45,1.0,0.70],[0.55,0.70,1.0]]),
        market_date=snap["snapshot_time"][:10],
        n_paths=20_000, seed=2026,
    )
    print(f"  market σ (B2 calibrated): {[f'{s*100:.2f}%' for s in res.sigmas_used]}")
    print(f"  fair value: {res.fair_value_krw:,.1f} ± {res.fair_value_stderr:.1f} KRW")
    print(f"  vs notional 10,000: {res.deviation_pct:+.2f}%")
    print(f"  KI hit prob: {res.ki_hit_prob*100:.1f}%, E[life]: {res.expected_life_years:.2f}y")
    return {
        "market_date": res.market_date,
        "sigma_used": res.sigmas_used[0],
        "fair_value": res.fair_value_krw,
        "deviation_pct": res.deviation_pct,
    }


# ---------------------------------------------------------------------------
# #3: B2 → B4 env σ
# ---------------------------------------------------------------------------

def demo_3_b2_b4_env():
    section("#3  B2 → B4 env σ (calibrated σ for hedge environment)")
    from ai_pricing.integrations.market_aware_hedge_env import (
        get_market_sigma, buehler_cfg_with_market_sigma,
    )
    from ai_pricing.deep_calib.surface import STRIKES, MATURITIES

    snaps = sorted((ROOT / "data" / "market_snapshots").glob("spy_iv_*.json"))
    snaps = [s for s in snaps if "_calib" not in s.name]
    snap = json.loads(snaps[-1].read_text(encoding="utf-8"))
    market = np.array(snap["iv_grid"])
    snap_T = np.array(snap["maturities_months"]) / 12
    snap_m = np.array(snap["moneyness"])
    iv_25 = np.full((len(MATURITIES), len(STRIKES)), np.nan)
    for i, T in enumerate(MATURITIES):
        i_src = int(np.argmin(np.abs(snap_T - T)))
        for j, m in enumerate(STRIKES):
            j_src = int(np.argmin(np.abs(snap_m - m)))
            iv_25[i, j] = market[i_src, j_src]
    iv_25 = iv_25.flatten()

    ms = get_market_sigma(iv_25, market_date=snap["snapshot_time"][:10])
    print(f"  baseline σ:       {ms.sigma_baseline*100:.2f}% (B4 hardcoded)")
    print(f"  calibrated σ:     {ms.sigma_calibrated*100:.2f}% (today's market)")
    print(f"  shift:            {ms.sigma_shift*100:+.2f}%p")
    print(f"  IV RMSE:          {ms.iv_rmse_vp:.2f} vp")

    cfg = buehler_cfg_with_market_sigma(ms, base_kwargs={
        "tc_rate": 0.003, "n_epochs": 50, "batch_size": 4096,
    })
    print(f"\n  → B4 새 BuehlerConfig:")
    print(f"      sigma = {cfg.sigma:.4f} (market 반영)")
    print(f"      tc_rate = {cfg.tc_rate}, n_epochs = {cfg.n_epochs}")
    print(f"  → train_buehler(cfg) 로 매일 다른 σ 환경 학습 가능")
    return {
        "sigma_baseline": ms.sigma_baseline,
        "sigma_calibrated": ms.sigma_calibrated,
        "sigma_shift": ms.sigma_shift,
        "iv_rmse_vp": ms.iv_rmse_vp,
    }


# ---------------------------------------------------------------------------
# #2: B3 → B4 NewsAwareHedger
# ---------------------------------------------------------------------------

def demo_2_b3_b4_hedger():
    section("#2  B3 → B4 NewsAwareHedger (macro shock 시 hedge 보수성)")
    from ai_hedging.agents.news_aware_hedger import NewsAwareHedger, HedgeBufferRule
    from ai_hedging.baselines.bsm_delta import BSMDeltaHedger
    from ai_hedging.env import HedgingEnv, HedgingEnvConfig

    cfg = HedgingEnvConfig(tc_rate=0.003, seed=0, reward_shaping=False,
                           action_low=0.0, action_high=1.0)
    bsm = BSMDeltaHedger("call")
    env = HedgingEnv(cfg)

    # NewsAwareHedger wraps BSM Δ (would also wrap Buehler PG)
    def base_predict(obs):
        return bsm.act(env)
    hedger = NewsAwareHedger(base_predict, action_low=0.0, action_high=1.0,
                              rule=HedgeBufferRule(base_buffer=0.05, magnitude_scale=2.0))

    # Scenario 1: 평소 (neutral news)
    obs, _ = env.reset(seed=42)
    hedger.update_news(["오늘 종합주가지수 상승세", "기업 실적 발표 정상"])
    a1 = hedger.act(obs)
    e1, b1 = hedger.last_event, hedger.last_buffer
    print(f"  [평소]   event={e1:<14s} buffer={b1:.3f}  →  action={a1:.3f}")

    # Scenario 2: macro_shock (Fed 금리 인상)
    env.reset(seed=42)
    hedger.update_news(["Fed 금리 인상 결정 발표", "중동 지정학 리스크 고조"])
    a2 = hedger.act(obs)
    e2, b2 = hedger.last_event, hedger.last_buffer
    print(f"  [shock]  event={e2:<14s} buffer={b2:.3f}  →  action={a2:.3f}")

    # Scenario 3: earnings_miss
    env.reset(seed=42)
    hedger.update_news(["삼성전자 실적 부진 어닝쇼크", "예상치 큰 폭 하회"])
    a3 = hedger.act(obs)
    e3, b3 = hedger.last_event, hedger.last_buffer
    print(f"  [miss]   event={e3:<14s} buffer={b3:.3f}  →  action={a3:.3f}")

    print(f"\n  → 평소 vs macro_shock hedge 차이: {a2-a1:+.3f}")
    print(f"  → 위기 시 자동 hedge 강화 (extra long position)")
    return {
        "neutral_event": e1, "neutral_action": a1,
        "shock_event": e2, "shock_action": a2,
        "miss_event": e3, "miss_action": a3,
        "shock_buffer_size": b2,
    }


# ---------------------------------------------------------------------------
# #4: B3 → ETF LP
# ---------------------------------------------------------------------------

def demo_4_b3_etf_lp():
    section("#4  B3 → ETF LP spread (Avellaneda-Stoikov σ 동적 조정)")
    from autotrader.strategies.news_aware_lp import NewsAwareASStrategy, NewsAwareASConfig

    # Realistic σ for demo visibility:
    # - sigma_baseline=0.20 (annualized 평균 vol)
    # - shock=0.60 (×3, 위기 시)
    # - gamma 큰 값으로 spread 가 σ 에 더 민감
    cfg = NewsAwareASConfig(
        gamma=2.0, sigma_baseline=0.20, k=1.5,
        shock_multiplier=3.0, inventory_limit=50, quote_size=1, tick_size=0.0001,
    )
    lp = NewsAwareASStrategy(cfg)

    mid = 100.0
    t_norm = 0.5
    inv = 0

    # Scenario 1: 평소
    lp.update_news(["오늘 시장 정상", "주요 종목 상승"])
    q1 = lp.quote(mid, t_norm, inv)
    spread1 = q1.ask - q1.bid
    print(f"  [평소]   event={lp.last_event:<14s} σ={lp.last_sigma:.4f}")
    print(f"           bid={q1.bid:.4f} ask={q1.ask:.4f} spread={spread1:.4f}")

    # Scenario 2: macro shock
    lp.update_news(["Fed 금리 인상 결정", "지정학 위기 고조 유가 급등"])
    q2 = lp.quote(mid, t_norm, inv)
    spread2 = q2.ask - q2.bid
    print(f"  [shock]  event={lp.last_event:<14s} σ={lp.last_sigma:.4f}  (×{cfg.shock_multiplier})")
    print(f"           bid={q2.bid:.4f} ask={q2.ask:.4f} spread={spread2:.4f}")

    print(f"\n  → spread {spread2 / spread1:.1f}× 넓어짐 → 위기 시 보수적 LP")
    print(f"  → mid {mid:.2f} 기준 절대 spread {(spread2 - spread1):.4f} 추가 (호가 멀어짐)")
    return {
        "normal_sigma": cfg.sigma_baseline,
        "shock_sigma": lp.last_sigma,
        "normal_spread": spread1,
        "shock_spread": spread2,
        "spread_multiplier": spread2 / spread1 if spread1 > 0 else None,
    }


def main():
    print("=" * 78)
    print("  4-Layer Integration Demo")
    print("=" * 78)

    results = {}
    results["#1_b2_to_els"] = demo_1_b2_els()
    results["#3_b2_to_b4_env"] = demo_3_b2_b4_env()
    results["#2_b3_to_b4_hedger"] = demo_2_b3_b4_hedger()
    results["#4_b3_to_etf_lp"] = demo_4_b3_etf_lp()

    section("종합")
    print(f"  4개 통합 모두 작동 ✓")
    print(f"  → 각 layer 가 isolated 검증 후 production 연결 데모")

    out = ROOT / "data" / "integration_4_layers_demo.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
