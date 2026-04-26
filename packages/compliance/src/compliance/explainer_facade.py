"""2-tier Explainability — Customer 자연어 + Regulator 수치 raw.

AI 기본법 회색지대 자발 baseline 의 4번째 모듈 (audit_log + deviation_monitor +
HITL gate 에 이어). 한 결정 = 두 층:

  customer 층 (~80자 자연어):
    "한화 ELS NAV 10,217원, 발행가 대비 +2.2%, 평균 만기 1.5년"
    → 영업/PB/고객 화면용

  regulator 층 (raw dict):
    {model: "MC_ELS_step_down", n_paths: 50000, sigma: [...], ki_prob: 0.14, ...}
    → 감사관 / 내부 통제 / 사후 재현용

설계 원칙:
  1. **결정론적** — LLM 호출 없음, template + format string. 감사 시 동일 결정 →
     동일 자연어 보장.
  2. **Plug-in** — 기존 4 audit_log call site 무수정. facade 만 추가 사용.
  3. **확장 가능** — facade.register(model, stage, fn) 한 줄로 신규 모델 추가.
     추후 추천 시스템 (PLE distill 등) 같은 자리에 끼움.
  4. **Fallback** — unknown (model, stage) 들어와도 generic template 으로 안전 실패.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from compliance.audit_log import DecisionEntry


@dataclass
class TwoTierExplanation:
    decision_id: str           # timestamp + model 조합 (audit log key)
    model_name: str
    stage: str
    customer_layer: str        # 자연어 ~80자
    regulator_layer: dict      # raw entry dict (regulator 가 보는 모든 수치)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "model_name": self.model_name,
            "stage": self.stage,
            "customer_layer": self.customer_layer,
            "regulator_layer": self.regulator_layer,
        }


# Type alias — template 함수 시그니처
TemplateFn = Callable[[DecisionEntry], str]


# -----------------------------------------------------------------------------
# 5 기본 templates — 현 audit log 의 (model_name, stage) 페어 커버
# -----------------------------------------------------------------------------

def _safe(v: Any, default: str = "?") -> str:
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4f}"
    return str(v)


def _pct(v: float | None, default: str = "?") -> str:
    if v is None:
        return default
    return f"{v:+.2f}%"


def template_els_morning_mtm(e: DecisionEntry) -> str:
    """MC_ELS_step_down / morning_mtm — 한화 ELS NAV."""
    output = e.output if isinstance(e.output, dict) else {}
    price = output.get("price_krw") or output.get("fair_value_krw") or e.output
    ki = output.get("ki_hit_prob")
    life = output.get("expected_life_years")

    bits = [f"한화 ELS NAV {_safe(price)}원"]
    if e.deviation_pct is not None:
        bits.append(f"발행가 대비 {_pct(e.deviation_pct)}")
    if life is not None:
        bits.append(f"평균 만기 {_safe(life)}년")
    if ki is not None:
        bits.append(f"KI 확률 {ki*100:.1f}%")
    return ". ".join(bits) + "."


def template_nn_5method(e: DecisionEntry) -> str:
    """NN_Pricer_log_500k / 5method_consistency — NN vs BSM 일관성."""
    output = e.output if isinstance(e.output, dict) else {}
    nn = output.get("call_price") or output.get("nn") or e.output
    bsm = e.reference_value if e.reference_value is not None else "?"
    inputs = e.inputs or {}
    sk_str = ""
    if "S" in inputs and "K" in inputs:
        sk_str = f"S={inputs['S']:.0f}/K={inputs['K']:.0f} "
    bits = [f"{sk_str}NN 가격 {_safe(nn)} vs BSM {_safe(bsm)}"]
    if e.deviation_pct is not None:
        verdict = "일관성 유지" if abs(e.deviation_pct) < 5 else "편차 주의"
        bits.append(f"차이 {_pct(e.deviation_pct)} ({verdict})")
    return ". ".join(bits) + "."


def template_buehler_hedge(e: DecisionEntry) -> str:
    """Buehler_PG_CVaR_historical / hedge_decision — 헤지 비율 결정."""
    output = e.output if isinstance(e.output, dict) else {}
    hedge = output.get("hedge_ratio") or output.get("hedge") or e.output
    bsm_delta = e.reference_value if e.reference_value is not None else "?"
    inputs = e.inputs or {}
    sigma = inputs.get("sigma")

    bits = [f"오늘 헤지 비율 {_safe(hedge)}"]
    if bsm_delta != "?":
        bits.append(f"BSM Δ {_safe(bsm_delta)} 대비 {_pct(e.deviation_pct)}")
    if sigma is not None:
        bits.append(f"변동성 {sigma*100:.1f}% 환경 반영")
    if e.hitl_required:
        bits.append("HITL 검토 요청")
    return ". ".join(bits) + "."


def template_as_lp_quote(e: DecisionEntry) -> str:
    """Avellaneda_Stoikov / lp_quote — 양방향 호가 + 재고."""
    output = e.output if isinstance(e.output, dict) else {}
    bid = output.get("bid")
    ask = output.get("ask")
    spread = output.get("spread")
    inputs = e.inputs or {}
    inv = inputs.get("inventory")
    inv_lim = e.reference_value if e.reference_value is not None else "?"
    gamma = inputs.get("gamma")

    bits = [f"양방향 호가 bid {_safe(bid)} / ask {_safe(ask)}"]
    if spread is not None:
        bits.append(f"spread {_safe(spread)}")
    if inv is not None:
        bits.append(f"재고 {inv}/{inv_lim}")
    if gamma is not None:
        bits.append(f"γ={gamma}")
    if e.deviation_breach:
        bits.append("재고 한도 임계 도달 — 보수적 호가")
    return ", ".join(bits) + "."


def template_news_classifier(e: DecisionEntry) -> str:
    """News_classifier / news_monitor — placeholder (현재 audit log 미사용)."""
    output = e.output if isinstance(e.output, dict) else {}
    event = output.get("event") or "neutral"
    delta_iv = output.get("delta_iv_pct") or output.get("iv_shift")
    headline = (e.inputs or {}).get("headline", "")
    if isinstance(headline, str) and len(headline) > 40:
        headline = headline[:40] + "..."
    bits = [f"뉴스 이벤트 분류: {event}"]
    if headline:
        bits.append(f'"{headline}"')
    if delta_iv is not None:
        bits.append(f"변동성 {_pct(delta_iv)} 보정")
    return ". ".join(bits) + "."


def template_generic_fallback(e: DecisionEntry) -> str:
    """unknown (model, stage) 일 때 generic."""
    out = _safe(e.output)
    if len(out) > 50:
        out = out[:50] + "..."
    bits = [f"{e.model_name} 모델이 {e.stage} 단계에서 결과 산출"]
    if e.reference_value is not None:
        bits.append(f"기준 {_safe(e.reference_value)} 대비 {_pct(e.deviation_pct)}")
    if e.hitl_required:
        bits.append("HITL 검토 권장")
    return " — ".join(bits) + "."


DEFAULT_TEMPLATES: dict[tuple[str, str], TemplateFn] = {
    ("MC_ELS_step_down", "morning_mtm"): template_els_morning_mtm,
    ("NN_Pricer_log_500k", "5method_consistency"): template_nn_5method,
    ("Buehler_PG_CVaR_historical", "hedge_decision"): template_buehler_hedge,
    ("Avellaneda_Stoikov", "lp_quote"): template_as_lp_quote,
    ("News_classifier", "news_monitor"): template_news_classifier,
}


# -----------------------------------------------------------------------------
# Facade
# -----------------------------------------------------------------------------


@dataclass
class ExplainerFacade:
    """기존 audit_log entry → 2-tier 자연어 변환 plug-in.

    Usage:
        facade = ExplainerFacade()
        explanation = facade.explain(entry)
        print(explanation.customer_layer)        # 자연어
        print(explanation.regulator_layer)       # raw dict

        # 신규 model 추가
        facade.register("PLE_recommender", "recommend",
                         lambda e: f"고객 성향 {e.inputs['risk']} → ...")
    """

    templates: dict[tuple[str, str], TemplateFn] = field(
        default_factory=lambda: dict(DEFAULT_TEMPLATES)
    )
    fallback: TemplateFn = template_generic_fallback

    def register(self, model_name: str, stage: str, fn: TemplateFn) -> None:
        """신규 (model, stage) → template 등록. 기존 키는 덮어씀."""
        self.templates[(model_name, stage)] = fn

    def explain(self, entry: DecisionEntry) -> TwoTierExplanation:
        """단일 entry → 2-tier."""
        # 이미 customer_layer 채워져 있으면 그대로 신뢰 (사용자가 explicit 작성한 경우)
        if entry.customer_layer:
            cust = entry.customer_layer
        else:
            fn = self.templates.get((entry.model_name, entry.stage), self.fallback)
            try:
                cust = fn(entry)
            except Exception as e:
                # template 실패해도 audit 흐름 깨지면 안 됨 — 최후 fallback
                cust = f"[explain error] {entry.model_name}/{entry.stage}: {type(e).__name__}"

        return TwoTierExplanation(
            decision_id=f"{entry.timestamp}__{entry.model_name}",
            model_name=entry.model_name,
            stage=entry.stage,
            customer_layer=cust,
            regulator_layer={
                "timestamp": entry.timestamp,
                "model_name": entry.model_name,
                "stage": entry.stage,
                "inputs": entry.inputs,
                "output": entry.output,
                "reference_model": entry.reference_model,
                "reference_value": entry.reference_value,
                "deviation_pct": entry.deviation_pct,
                "deviation_threshold_pct": entry.deviation_threshold_pct,
                "deviation_breach": entry.deviation_breach,
                "hitl_required": entry.hitl_required,
                "hitl_reason": entry.hitl_reason,
                "extra": entry.extra,
            },
        )

    def explain_all(self, entries: list[DecisionEntry]) -> list[TwoTierExplanation]:
        return [self.explain(e) for e in entries]

    def known_pairs(self) -> list[tuple[str, str]]:
        return list(self.templates.keys())


__all__ = [
    "TwoTierExplanation",
    "ExplainerFacade",
    "TemplateFn",
    "DEFAULT_TEMPLATES",
    # 개별 templates 도 export — 추천 시스템 등에서 참고용
    "template_els_morning_mtm",
    "template_nn_5method",
    "template_buehler_hedge",
    "template_as_lp_quote",
    "template_news_classifier",
    "template_generic_fallback",
]
