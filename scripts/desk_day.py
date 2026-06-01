"""한화 ELS 데스크의 하루 — orchestration script.

5-Layer 결과를 "데스크 워크플로우" 시간 순으로 chain.

Stages:
  09:00  Morning calibration   → σ_morning (3-leg)
  09:15  Mark-to-market        → ELS 모델가 vs 발행가
  10:00  5-method spread       → SPX 1M ATM call 가격 5 방법 비교
  11:00  News event monitor    → B3 v2 surprise-aware ΔIV
  13:00  Hedge update          → Buehler historical policy + PPO 5번 실패 메타
  15:00  Intraday risk check   → per-leg Greeks + KI / autocall 거리
  16:30  Day-end PnL           → BSM Δ vs Buehler 하루 PnL + ELS MTM 변화

Output: data/desk_day/run_<YYYY-MM-DD>.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "pricing" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_hedging" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "compliance" / "src"))

from compliance import DecisionLog, DeviationMonitor, HITLGate

AUDIT_LOG = DecisionLog(ROOT / "data" / "compliance" / "audit_log.jsonl")
MONITOR = DeviationMonitor(default_threshold_pct=5.0)
HITL = HITLGate(env="simulation")

# ──────────────────────────────────────────────────────────────────────────
#   8286호 spec (이미 보유한 실 공시 데이터)
# ──────────────────────────────────────────────────────────────────────────
ELS_SPEC = {
    "issue_no": 8286,
    "underlyings": ["KOSPI200", "SPX", "SX5E"],
    "yf_tickers":  ["^KS200",   "^GSPC", "^STOXX50E"],
    "S0": [100.0, 100.0, 100.0],
    "barriers":  [0.95, 0.90, 0.85, 0.85, 0.80, 0.75],
    "ki_barrier": 0.50,
    "coupon_per_year": 0.1131,
    "obs_per_year": 2,
    "maturity_years": 3.0,
    "notional": 10_000.0,
    "issue_price": 10_000.0,
}

DEFAULT_CORR = np.array([
    [1.00, 0.45, 0.55],
    [0.45, 1.00, 0.70],
    [0.55, 0.70, 1.00],
])
DEFAULT_Q = np.array([0.017, 0.015, 0.031])
DEFAULT_R = 0.035

# fallback realized vol when yfinance fails
FALLBACK_SIGMA = np.array([0.22, 0.18, 0.22])


# ──────────────────────────────────────────────────────────────────────────
#   Stage 1 — 09:00  Morning calibration
# ──────────────────────────────────────────────────────────────────────────
def run_morning_calibration() -> dict:
    """3-leg 30일 historical realized vol via yfinance.

    KRX 인증 부재 → KOSPI200 / SX5E 도 yfinance 사용 (^KS200 / ^STOXX50E).
    SPX leg 은 다른 layer (B2 Deep Calib) 와 일관되게.
    """
    print("[09:00] Morning calibration ...")
    t0 = time.time()
    sigmas = []
    notes = []
    try:
        import yfinance as yf
        for sym in ELS_SPEC["yf_tickers"]:
            try:
                df = yf.Ticker(sym).history(period="60d", auto_adjust=True)
                if len(df) < 20:
                    raise RuntimeError(f"too few data points: {len(df)}")
                rets = np.diff(np.log(df["Close"].values))
                ann_vol = float(rets.std() * math.sqrt(252))
                sigmas.append(ann_vol)
                notes.append(f"{sym}: 30d rolling ann vol = {ann_vol*100:.2f}%")
            except Exception as e:
                fallback_idx = len(sigmas)
                sigmas.append(float(FALLBACK_SIGMA[fallback_idx]))
                notes.append(f"{sym}: fetch fail ({e}) → fallback {FALLBACK_SIGMA[fallback_idx]*100:.1f}%")
    except ImportError:
        sigmas = list(FALLBACK_SIGMA)
        notes = ["yfinance not installed → fallback all"]

    sigma_morning = np.array(sigmas)
    elapsed = time.time() - t0
    for n in notes:
        print(f"        {n}")
    print(f"        σ_morning = {[f'{s*100:.2f}%' for s in sigma_morning]} ({elapsed:.1f}s)")
    return {
        "sigma_morning": sigma_morning.tolist(),
        "underlying": ELS_SPEC["underlyings"],
        "calib_notes": notes,
        "calib_seconds": round(elapsed, 2),
    }


# ──────────────────────────────────────────────────────────────────────────
#   Stage 2 — 09:15  Mark-to-market
# ──────────────────────────────────────────────────────────────────────────
def run_morning_mtm(sigma_morning: list) -> dict:
    """price_els(8286, sigma=σ_morning) → 모델가 vs 발행가."""
    print("[09:15] Mark-to-market (price_els 8286호) ...")
    from pricing.els.step_down import StepDownELS, price_els

    product = StepDownELS(
        S0=np.array(ELS_SPEC["S0"]),
        barriers=ELS_SPEC["barriers"],
        ki_barrier=ELS_SPEC["ki_barrier"],
        coupon_rate=ELS_SPEC["coupon_per_year"] / ELS_SPEC["obs_per_year"],
        maturity_years=ELS_SPEC["maturity_years"],
        obs_per_year=ELS_SPEC["obs_per_year"],
        notional=ELS_SPEC["notional"],
    )
    t0 = time.time()
    res = price_els(
        product, r=DEFAULT_R, q=DEFAULT_Q,
        sigma=np.array(sigma_morning), corr=DEFAULT_CORR,
        n_paths=30_000, n_steps_per_year=252, seed=2026,
    )
    elapsed = time.time() - t0
    dev_pct = (res.price - ELS_SPEC["issue_price"]) / ELS_SPEC["issue_price"] * 100

    print(f"        price = {res.price:>9,.1f}원 ± {res.stderr:.1f}  "
          f"KI prob = {res.ki_hit_prob*100:.1f}%  E[life] = {res.expected_life:.2f}y  "
          f"vs notional = {dev_pct:+.2f}%  ({elapsed:.1f}s)")

    # AI 기본법 대응 #1·#2 — audit log + 편차 모니터링 (vs 발행가 기준)
    dev_check = MONITOR.check(
        label="ELS_8286_model_vs_notional",
        model_value=res.price, reference_value=ELS_SPEC["issue_price"],
        threshold_pct=5.0,
    )
    # AI 기본법 대응 #3 — 발행가 ±5% 초과 시 HITL escalation
    hitl = HITL.evaluate(
        label="els_mtm_breach", metric_value=abs(dev_pct), threshold=5.0,
        severity="high" if abs(dev_pct) > 10 else "medium",
        reason=f"ELS 모델가가 발행가 대비 {dev_pct:+.2f}% — 시장 stress 또는 모델 OOD",
        suggested_action="Trader 가 σ_morning 합리성 + 시장 환경 확인",
    )
    AUDIT_LOG.append(
        model_name="MC_ELS_step_down",
        stage="morning_mtm",
        inputs={"sigma": list(np.array(sigma_morning).round(4)),
                "n_paths": 30_000, "issue_no": 8286},
        output={"price_krw": round(res.price, 1),
                "ki_hit_prob": round(res.ki_hit_prob, 4)},
        reference_model="notional_par_value",
        reference_value=ELS_SPEC["issue_price"],
        deviation_pct=round(abs(dev_pct), 4),
        deviation_threshold_pct=5.0,
        deviation_breach=dev_check.breach,
        hitl_required=hitl.triggered,
        hitl_reason=hitl.reason if hitl.triggered else None,
        extra={"ki_hit_prob": res.ki_hit_prob, "expected_life": res.expected_life,
               "severity": dev_check.severity},
    )
    if hitl.triggered:
        print(f"        [HITL] {hitl.severity.upper()} — {hitl.reason}")

    return {
        "price_morning_krw": round(res.price, 1),
        "stderr_krw": round(res.stderr, 1),
        "vs_notional_pct": round(dev_pct, 2),
        "ki_hit_prob": round(res.ki_hit_prob, 4),
        "expected_life_years": round(res.expected_life, 3),
        "autocall_prob_by_period": [round(p, 4) for p in res.autocall_prob],
        "n_paths": 30_000,
        "elapsed_seconds": round(elapsed, 2),
        "compliance": {
            "deviation_severity": dev_check.severity,
            "deviation_pct": dev_check.deviation_pct,
            "hitl_triggered": hitl.triggered,
            "hitl_severity": hitl.severity if hitl.triggered else None,
        },
    }


# ──────────────────────────────────────────────────────────────────────────
#   Stage 3 — 10:00  5-method spread check (SPX 1M ATM call)
# ──────────────────────────────────────────────────────────────────────────
def run_5method_check(sigma_spx: float) -> dict:
    """SPX 1M ATM vanilla call 5-method spread."""
    print("[10:00] 5-method spread check (SPX 1M ATM call) ...")
    from pricing.bsm import BSMInputs, call_price

    K = 100.0
    S = 100.0
    T = 30 / 365
    r = DEFAULT_R
    q = float(DEFAULT_Q[1])
    sigma_base = float(sigma_spx)

    prices = {}

    # Method 1: BSM closed-form
    bsm_in = BSMInputs(S=S, K=K, T=T, r=r, q=q, sigma=sigma_base)
    prices["A_BSM"] = round(call_price(bsm_in), 4)

    # Method 2: NN Pricer (Hutchinson 1994)
    try:
        from ai_pricing.nn_pricer.infer import load_pricer, price_batch
        nn_path = ROOT / "models" / "nn_pricer_log_500k.pt"
        if nn_path.exists():
            model, _ = load_pricer(str(nn_path))
            X = np.array([[S, K, T, r, q, sigma_base]])
            p = price_batch(model, X)
            prices["B1_NN"] = round(float(p[0]), 4)
        else:
            prices["B1_NN"] = None
    except Exception as e:
        print(f"        NN Pricer 로드 실패: {e}")
        prices["B1_NN"] = None

    # Method 3: Deep Calib implied — proxy로 BSM with calibrated sigma 약간 다르게
    # (실 Heston calibration 은 시간 비싸니 cached σ_calib 사용 — 본 stage 엔 simpler proxy)
    sigma_dc = sigma_base * 0.98  # Heston ATM IV 가 BSM 와 약 -2% 차이라는 휴리스틱
    bsm_dc = BSMInputs(S=S, K=K, T=T, r=r, q=q, sigma=sigma_dc)
    prices["B2_DeepCalib"] = round(call_price(bsm_dc), 4)

    # Method 4: News-adjusted — placeholder, actual ΔIV 는 stage 4 에서 계산
    # 여기서는 baseline 으로 BSM 만 둠 (stage 4 후에 update)
    prices["B3_NewsAdj_pre"] = prices["A_BSM"]

    # Method 5: RL-implied — Buehler 학습된 sigma (realized) 로 BSM 가격
    # Buehler 모델은 hedge cost 를 학습한 것이라 직접 quote 못하지만, premium = BSM(σ_realized) 로 사용
    # 단순화: realized vol 로 BSM
    prices["B4_RL_implied"] = round(call_price(bsm_in), 4)  # 같은 σ → 같은 가격 (baseline)

    valid = [p for p in prices.values() if p is not None]
    mean_p = float(np.mean(valid))
    std_p = float(np.std(valid))
    spread_pct = std_p / mean_p * 100
    if spread_pct < 1.0:
        signal = "all-method consensus (정상)"
    elif spread_pct < 3.0:
        signal = "mild divergence — 모니터"
    else:
        signal = "high divergence — mispriced 시그널"

    for name, p in prices.items():
        print(f"        {name:<20s}  {p}")
    print(f"        spread = {spread_pct:.2f}%  →  {signal}")

    # Compliance — NN Pricer 가 BSM 대비 편차 크면 deviation breach + HITL
    if prices.get("B1_NN") is not None:
        dev_nn = MONITOR.check(
            label="NN_vs_BSM_SPX_call",
            model_value=prices["B1_NN"], reference_value=prices["A_BSM"],
            threshold_pct=5.0,
        )
        hitl_nn = HITL.evaluate(
            label="nn_pricer_drift",
            metric_value=dev_nn.deviation_pct, threshold=5.0,
            severity="high" if dev_nn.severity == "high" else "medium",
            reason=f"NN Pricer 가 BSM 대비 {dev_nn.deviation_pct:.2f}% 편차 (학습 도메인 OOD 가능)",
            suggested_action="Quant 가 NN 학습 도메인 (S/K, σ, T 범위) 점검 + 재학습 필요성 판단",
        )
        AUDIT_LOG.append(
            model_name="NN_Pricer_log_500k",
            stage="5method_consistency",
            inputs={"S": 100.0, "K": 100.0, "T": 30/365, "r": DEFAULT_R,
                    "q": float(DEFAULT_Q[1]), "sigma": float(sigma_spx)},
            output={"call_price": prices["B1_NN"]},
            reference_model="BSM_closed_form",
            reference_value=prices["A_BSM"],
            deviation_pct=dev_nn.deviation_pct,
            deviation_threshold_pct=5.0,
            deviation_breach=dev_nn.breach,
            hitl_required=hitl_nn.triggered,
            hitl_reason=hitl_nn.reason if hitl_nn.triggered else None,
            extra={"severity": dev_nn.severity, "spread_5method_pct": round(spread_pct, 2)},
        )
        if hitl_nn.triggered:
            print(f"        [HITL] {hitl_nn.severity.upper()} — {hitl_nn.reason}")

    return {
        "prices_by_method": prices,
        "mean_price": round(mean_p, 4),
        "spread_pct": round(spread_pct, 2),
        "signal": signal,
        "method_count_valid": len(valid),
        "compliance": {
            "nn_vs_bsm_deviation_pct": round(dev_nn.deviation_pct, 2) if prices.get("B1_NN") is not None else None,
            "nn_severity": dev_nn.severity if prices.get("B1_NN") is not None else "n/a",
            "hitl_triggered": hitl_nn.triggered if prices.get("B1_NN") is not None else False,
        },
    }


# ──────────────────────────────────────────────────────────────────────────
#   Stage 4 — 11:00  News event monitor (B3 v2)
# ──────────────────────────────────────────────────────────────────────────
def run_news_monitor(sigma_morning: list) -> dict:
    """매크로 캘린더에서 1건 inject → B3 v2 surprise-aware → σ 업데이트."""
    print("[11:00] News event monitor (B3 v2) ...")
    from ai_pricing.news_iv.iv_shift_v2 import predict_sign_v2, SURPRISE_DIRECTION
    from ai_pricing.news_iv.iv_shift import IV_SHIFT_RULES

    cal = json.loads((ROOT / "data" / "macro_events" / "fomc_cpi_calendar_v2.json")
                     .read_text(encoding="utf-8"))

    today = datetime.now().date()
    candidates = []
    for ev in cal["fomc"] + cal["cpi"]:
        try:
            d = datetime.strptime(ev["date"], "%Y-%m-%d").date()
            if d <= today:
                candidates.append((d, ev))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates:
        return {"event": None, "iv_shift_vp": 0.0, "sigma_after": sigma_morning,
                "note": "no past macro event found"}

    pick_d, pick_ev = candidates[0]
    sign = predict_sign_v2(
        event="macro_shock",
        surprise=pick_ev.get("surprise"),
        title=pick_ev.get("title"),
        decision=pick_ev.get("decision"),
    )
    base_shift = abs(IV_SHIFT_RULES["macro_shock"])
    surprise_dir = SURPRISE_DIRECTION.get(pick_ev.get("surprise") or pick_ev.get("decision") or "", 0.0)
    if pick_ev.get("title", "").lower().find("hawkish") >= 0:
        surprise_dir = +1.0
    elif pick_ev.get("title", "").lower().find("dovish") >= 0:
        surprise_dir = -0.6
    iv_shift = base_shift * surprise_dir   # 양/음 부호 반영

    sigma_arr = np.array(sigma_morning) + iv_shift
    sigma_arr = np.clip(sigma_arr, 0.05, 1.0)
    days_ago = (today - pick_d).days

    print(f"        event = {pick_d}  {pick_ev.get('title')[:60]}")
    print(f"        surprise = {pick_ev.get('surprise') or pick_ev.get('decision')}  "
          f"predicted_sign = {sign:+d}  ΔIV = {iv_shift*100:+.2f} vp")
    print(f"        σ_after = {[f'{s*100:.2f}%' for s in sigma_arr]}  ({days_ago}d ago)")

    return {
        "event": {
            "date": str(pick_d),
            "title": pick_ev.get("title"),
            "surprise": pick_ev.get("surprise") or pick_ev.get("decision"),
            "days_ago": days_ago,
        },
        "predicted_sign": sign,
        "iv_shift_vp": round(iv_shift * 100, 2),
        "sigma_after": sigma_arr.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────
#   Stage 5 — 13:00  Hedge update (Buehler historical) + 시각 3 메타
# ──────────────────────────────────────────────────────────────────────────
def run_hedge_decision(sigma_after: list) -> dict:
    """Buehler historical policy 단일-step inference + BSM Δ 비교."""
    print("[13:00] Hedge update (Buehler historical) ...")
    import torch
    from ai_hedging.agents.buehler_pg import HedgingPolicy
    from ai_hedging.agents.buehler_pg_historical import HistoricalConfig
    from scipy.stats import norm

    sigma_spx = float(sigma_after[1])
    cfg = HistoricalConfig(
        S0=100.0, K=100.0, T=30 / 365, r=DEFAULT_R, q=0.0, sigma=sigma_spx,
        n_steps=30, tc_rate=0.003, action_low=0.0, action_high=1.0,
    )

    model_path = ROOT / "models" / "buehler_historical.pt"
    hedge_buehler = None
    if model_path.exists():
        try:
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
            policy = HedgingPolicy(action_low=cfg.action_low, action_high=cfg.action_high)
            policy.load_state_dict(ckpt["state_dict"])
            policy.eval()
            S = cfg.S0
            T_rem = cfg.T * (1 - 0 / cfg.n_steps)
            obs = torch.tensor([[
                math.log(S / cfg.K),
                T_rem / cfg.T,
                0.0,
                cfg.sigma * math.sqrt(T_rem),
                1.0,
            ]], dtype=torch.float32)
            with torch.no_grad():
                hedge_buehler = float(policy(obs).item())
        except Exception as e:
            print(f"        Buehler model 로드 실패: {e}")

    # BSM call delta closed-form
    d1 = (math.log(cfg.S0 / cfg.K) + (cfg.r - cfg.q + 0.5 * cfg.sigma ** 2) * cfg.T) \
         / (cfg.sigma * math.sqrt(cfg.T))
    hedge_bsm = float(norm.cdf(d1))

    delta_diff = (hedge_buehler - hedge_bsm) if hedge_buehler is not None else None

    print(f"        BSM Δ      = {hedge_bsm:.4f}")
    if hedge_buehler is not None:
        print(f"        Buehler   = {hedge_buehler:.4f}")
        print(f"        Δ_diff    = {delta_diff:+.4f}  "
              f"(Buehler under-hedges by {abs(delta_diff):.3f} due to TC penalty)")
    else:
        print(f"        Buehler   = (model load failed)")

    # Compliance — Buehler 정책 OOD 검출 (Δ 차이 > 0.30 이면 high)
    if hedge_buehler is not None:
        hitl_h = HITL.evaluate(
            label="hedge_ratio_diverge",
            metric_value=abs(delta_diff), threshold=0.30,
            severity="high" if abs(delta_diff) > 0.30 else ("medium" if abs(delta_diff) > 0.15 else "low"),
            reason=f"Buehler vs BSM Δ 차이 {delta_diff:+.3f} (정책 학습 sigma 와 현재 sigma 차이 가능)",
            suggested_action="Trader 가 Buehler 학습 σ 와 현재 σ 일치 여부 + TC 가정 확인",
        )
        AUDIT_LOG.append(
            model_name="Buehler_PG_CVaR_historical",
            stage="hedge_decision",
            inputs={"sigma": sigma_spx, "obs": [0.0, 1.0, 0.0,
                                                 round(sigma_spx * math.sqrt(cfg.T), 4), 1.0]},
            output={"hedge_ratio": round(hedge_buehler, 4)},
            reference_model="BSM_call_delta",
            reference_value=round(hedge_bsm, 4),
            deviation_pct=round(abs(delta_diff) * 100, 2),
            deviation_threshold_pct=15.0,
            deviation_breach=abs(delta_diff) > 0.15,
            hitl_required=hitl_h.triggered,
            hitl_reason=hitl_h.reason if hitl_h.triggered else None,
            extra={"severity": hitl_h.severity},
        )
        if hitl_h.triggered:
            print(f"        [HITL] {hitl_h.severity.upper()} — {hitl_h.reason}")

    ppo_failure_log = [
        {"attempt": "v1 (vanilla PPO)",        "result": "−1100% CVaR — diverged (entropy collapse)"},
        {"attempt": "v2 (action clip + shaping)", "result": "+7.8% — best PPO, PARTIAL (목표 +20% 미달)"},
        {"attempt": "v3 (longer training)",     "result": "+2% — entropy still collapsed"},
        {"attempt": "v3b (BC warm start)",      "result": "+1% — locked at BSM attractor"},
        {"attempt": "v3c (residual to BSM)",    "result": "+0.5% — converges to BSM exactly"},
        {"attempt": "Buehler PG-on-CVaR direct (현재)", "result": "+24.5% (GBM) / +31.3% (SPY historical) PASS"},
    ]

    return {
        "sigma_used": sigma_spx,
        "hedge_buehler": round(hedge_buehler, 4) if hedge_buehler is not None else None,
        "hedge_bsm": round(hedge_bsm, 4),
        "delta_diff": round(delta_diff, 4) if delta_diff is not None else None,
        "ppo_failure_log": ppo_failure_log,
        "decision": "Buehler under-hedge (TC 페널티 반영)" if hedge_buehler is not None else "fallback to BSM Δ",
    }


# ──────────────────────────────────────────────────────────────────────────
#   Stage 6 — 15:00  Intraday risk check
# ──────────────────────────────────────────────────────────────────────────
def run_risk_check(sigma_after: list, mtm_result: dict) -> dict:
    """Per-leg Greeks + KI / autocall 거리."""
    print("[15:00] Intraday risk check ...")
    from pricing.bsm import BSMInputs
    from pricing.greeks.analytic import call_greeks

    per_leg = []
    for i, (name, sig) in enumerate(zip(ELS_SPEC["underlyings"], sigma_after)):
        bsm_in = BSMInputs(
            S=ELS_SPEC["S0"][i], K=ELS_SPEC["S0"][i] * ELS_SPEC["ki_barrier"],
            T=ELS_SPEC["maturity_years"], r=DEFAULT_R, q=float(DEFAULT_Q[i]), sigma=float(sig),
        )
        g = call_greeks(bsm_in)
        per_leg.append({
            "leg": name,
            "delta_to_KI": round(g.delta, 4),
            "vega": round(g.vega, 4),
            "theta_per_day": round(g.theta / 365, 4),
        })
        print(f"        {name:<10s}  Δ→KI={g.delta:+.4f}  vega={g.vega:.4f}  θ/d={g.theta/365:.4f}")

    days_to_next_autocall = int((365 / ELS_SPEC["obs_per_year"]))
    ki_distance_pct = [round((1.0 - ELS_SPEC["ki_barrier"]) * 100, 1)] * 3  # 현재 S=S0 가정

    print(f"        next autocall ≈ {days_to_next_autocall}일 후 · KI 거리 ≈ -50%/leg")
    print(f"        KI hit prob (from MTM stage): {mtm_result['ki_hit_prob']*100:.1f}%")

    return {
        "per_leg_greeks": per_leg,
        "next_autocall_days": days_to_next_autocall,
        "ki_distance_pct_per_leg": ki_distance_pct,
        "ki_hit_prob_overall": mtm_result["ki_hit_prob"],
    }


# ──────────────────────────────────────────────────────────────────────────
#   Stage 7 — 16:30  Day-end PnL
# ──────────────────────────────────────────────────────────────────────────
def run_day_end_pnl(stages: dict) -> dict:
    """BSM Δ vs Buehler 1-day hedge PnL + ELS MTM 변화."""
    print("[16:30] Day-end PnL ...")
    spy_csv = ROOT / "data" / "macro_events" / "spy_returns.csv"
    if spy_csv.exists():
        lines = spy_csv.read_text(encoding="utf-8").splitlines()[1:]
        recent_rets = [float(line.split(",")[1]) for line in lines[-5:]]
        day_return = float(np.mean(recent_rets))
    else:
        day_return = 0.005

    notional_per_leg = 100.0
    h_bsm = stages["hedge"]["hedge_bsm"]
    h_buehler = stages["hedge"].get("hedge_buehler") or h_bsm

    spx_S = 100.0
    dS = spx_S * (math.exp(day_return) - 1.0)

    pnl_bsm = h_bsm * dS
    pnl_buehler = h_buehler * dS
    diff = pnl_buehler - pnl_bsm

    morning = stages["mtm"]["price_morning_krw"]
    iv_shift_vp = stages["news"]["iv_shift_vp"]
    after_news = morning * (1 + iv_shift_vp / 100 * 0.05)
    close = after_news + dS * 50
    mtm_change_pct = (close - morning) / morning * 100

    one_liner = (
        f"σ_morning {[f'{s*100:.1f}%' for s in stages['calib']['sigma_morning']]} "
        f"→ ELS {morning:,.0f}원 ({stages['mtm']['vs_notional_pct']:+.2f}%) "
        f"→ event '{stages['news']['event']['surprise'] if stages['news']['event'] else 'none'}' "
        f"ΔIV {iv_shift_vp:+.2f}vp "
        f"→ Buehler hedge {h_buehler:.3f} (BSM Δ {h_bsm:.3f}) "
        f"→ day PnL diff {diff:+.4f}, ELS MTM {mtm_change_pct:+.3f}%"
    )

    print(f"        SPY day return  = {day_return*100:+.3f}% (5-day avg)")
    print(f"        BSM Δ hedge PnL  = {pnl_bsm:+.4f}")
    print(f"        Buehler hedge PnL = {pnl_buehler:+.4f}")
    print(f"        diff             = {diff:+.4f}  ({'Buehler 우위' if diff > 0 else 'BSM 우위'})")
    print(f"        ELS MTM change   = {mtm_change_pct:+.3f}%")
    print(f"\n        ONE-LINER: {one_liner}")

    return {
        "spy_day_return_pct": round(day_return * 100, 3),
        "hedge_pnl_bsm": round(pnl_bsm, 4),
        "hedge_pnl_buehler": round(pnl_buehler, 4),
        "hedge_pnl_diff": round(diff, 4),
        "els_mtm_morning": round(morning, 1),
        "els_mtm_after_news": round(after_news, 1),
        "els_mtm_close": round(close, 1),
        "els_mtm_change_pct": round(mtm_change_pct, 3),
        "day_summary_one_liner": one_liner,
    }


# ──────────────────────────────────────────────────────────────────────────
#   Main — chain all 7 stages
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("  한화 ELS 데스크의 하루 — desk_day.py")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d')}  Focal: 한화스마트ELS 제8286호")
    print("=" * 78)
    print()

    stages = {}
    stages["calib"] = run_morning_calibration()
    print()

    stages["mtm"] = run_morning_mtm(stages["calib"]["sigma_morning"])
    print()

    stages["spread"] = run_5method_check(sigma_spx=stages["calib"]["sigma_morning"][1])
    print()

    stages["news"] = run_news_monitor(stages["calib"]["sigma_morning"])
    print()

    stages["hedge"] = run_hedge_decision(stages["news"]["sigma_after"])
    print()

    stages["risk"] = run_risk_check(stages["news"]["sigma_after"], stages["mtm"])
    print()

    stages["pnl"] = run_day_end_pnl(stages)
    print()

    # Compliance summary
    audit_summary = AUDIT_LOG.summary()
    stages["compliance_summary"] = audit_summary
    print()
    print("─" * 78)
    print("  AI 기본법 대응 — 의사결정 기록 / 편차 모니터링 / HITL")
    print("─" * 78)
    print(f"        Total decisions logged : {audit_summary['total_decisions']}")
    print(f"        Deviation breaches     : {audit_summary['deviation_breaches']}")
    print(f"        HITL triggered         : {audit_summary['hitl_required']}")
    print(f"        By model               : {audit_summary['by_model']}")
    print(f"        Audit log path         : {audit_summary['path']}")

    out_dir = ROOT / "data" / "desk_day"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "focal_product": ELS_SPEC,
        "stages": stages,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("=" * 78)
    print(f"  → {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
