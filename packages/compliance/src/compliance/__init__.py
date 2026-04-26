"""AI 기본법 대응 — 의사결정 기록 / 편차 모니터링 / HITL gate / 2-tier 설명.

Korean AI Basic Law (인공지능 발전과 신뢰 기반 조성 등에 관한 법률) 대응 모듈.
시행 2026-01-22, 본 프로젝트는 회색지대 (시가평가는 환매가/발행가에 간접 영향).

Self-imposed safeguards (4 종):
  - audit_log.DecisionLog: 모든 AI 모델 의사결정 기록 (JSON Lines)
  - deviation_monitor.DeviationMonitor: 기준 모델(BSM/Heston)과 편차 threshold check
  - hitl_gate.HITLGate: 사람 검증 필요 결정 gate (high-severity 시 alert + flag)
  - explainer_facade.ExplainerFacade: customer 자연어 + regulator 수치 2-tier 설명
"""

from compliance.audit_log import DecisionLog, DecisionEntry
from compliance.deviation_monitor import DeviationMonitor, DeviationResult
from compliance.hitl_gate import HITLGate, HITLDecision
from compliance.explainer_facade import ExplainerFacade, TwoTierExplanation

__all__ = [
    "DecisionLog", "DecisionEntry",
    "DeviationMonitor", "DeviationResult",
    "HITLGate", "HITLDecision",
    "ExplainerFacade", "TwoTierExplanation",
]
