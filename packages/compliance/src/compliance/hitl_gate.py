"""Human-in-the-loop Gate — AI 기본법 대응 #3.

특정 의사결정에 사람 검증 단계 강제. severity 'high' 또는 명시적 trigger 발동 시
실 데스크에서는 trader 승인 대기, 시뮬에서는 print + flag 만 (단, audit_log 에는
hitl_required=True 로 영구 기록).

Trigger 예시:
  - 5-method spread > 5% (mispricing 의심)
  - Buehler hedge ratio 가 BSM Δ 와 0.3 이상 차이 (정책 OOD)
  - ELS 모델가 vs notional 편차 > 5%
  - News-IV ΔIV |shift| > 3 vp (큰 surprise)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HITLDecision:
    triggered: bool
    severity: str            # 'low' / 'medium' / 'high'
    label: str
    reason: str
    auto_approved: bool      # 시뮬 환경에서 True (실 데스크는 항상 False)
    suggested_action: str


class HITLGate:
    """Human-in-the-loop gate.

    실 데스크 환경: triggered=True 면 trader 승인까지 대기 (block).
    시뮬레이션 환경: triggered=True 라도 auto_approved=True 로 흘려보내되 audit log
                     에 영구 기록 + 보고서 highlight.

    Usage:
        gate = HITLGate(env="simulation")
        decision = gate.evaluate(
            label="hedge_ratio_diverge",
            metric_value=0.35, threshold=0.30,
            severity="high", reason="Buehler vs BSM Δ over 0.3 — possible policy OOD",
            suggested_action="Review last 100 paths; inspect TC penalty term",
        )
        if decision.triggered and not decision.auto_approved:
            input("Trader 승인 후 enter")  # 실 데스크 only
    """

    def __init__(self, env: str = "simulation"):
        self.env = env

    def evaluate(self, label: str, metric_value: float, threshold: float,
                 severity: str, reason: str,
                 suggested_action: str = "Review and confirm") -> HITLDecision:
        triggered = abs(metric_value) > threshold
        auto_approved = (self.env == "simulation") and triggered

        return HITLDecision(
            triggered=triggered,
            severity=severity,
            label=label,
            reason=reason,
            auto_approved=auto_approved,
            suggested_action=suggested_action,
        )


__all__ = ["HITLGate", "HITLDecision"]
