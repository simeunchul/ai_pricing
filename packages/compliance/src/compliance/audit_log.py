"""의사결정 기록 (Decision Audit Log) — AI 기본법 대응 #1.

Append-only JSON Lines log. 모든 AI 모델 의사결정 (NN Pricer 가격, Buehler hedge,
B3 IV shift 등) 을 timestamp + inputs + output + reference + deviation + HITL flag
와 함께 기록.

Format: data/compliance/audit_log.jsonl  (한 줄 = 한 의사결정)

용도:
  1. 사후 감사 (regulator inquiry 시 재현 가능)
  2. 편차 시계열 분석 (모델 drift 탐지)
  3. HITL escalation pattern (어떤 결정이 자주 사람 검증으로 갔는가)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DecisionEntry:
    timestamp: str
    model_name: str          # "NN_Pricer_log_500k", "Buehler_historical", "B3_v2_surprise", ...
    stage: str               # "morning_calib", "mtm", "5method", "hedge", ...
    inputs: dict[str, Any]   # 재현용 입력
    output: Any              # 모델 결과
    reference_model: str | None = None     # 비교 기준 (보통 BSM)
    reference_value: Any = None
    deviation_pct: float | None = None
    deviation_threshold_pct: float | None = None
    deviation_breach: bool = False
    hitl_required: bool = False
    hitl_reason: str | None = None
    customer_layer: str | None = None     # 자연어 설명 (선택, AI 기본법 회색지대 자발 baseline)
    extra: dict[str, Any] = field(default_factory=dict)


class DecisionLog:
    """Append-only audit log.

    Usage:
        log = DecisionLog("data/compliance/audit_log.jsonl")
        log.append(model_name="NN_Pricer", stage="quote", inputs={...}, output=1.97,
                   reference_model="BSM", reference_value=1.84,
                   deviation_pct=7.2, deviation_threshold_pct=5.0)
    """

    def __init__(self, path: str | Path = "data/compliance/audit_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, **kwargs) -> DecisionEntry:
        kwargs.setdefault("timestamp", datetime.now().isoformat())
        entry = DecisionEntry(**kwargs)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False, default=str) + "\n")
        return entry

    def read_all(self) -> list[DecisionEntry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(DecisionEntry(**d))
        return out

    def summary(self) -> dict:
        entries = self.read_all()
        n = len(entries)
        n_breach = sum(1 for e in entries if e.deviation_breach)
        n_hitl = sum(1 for e in entries if e.hitl_required)
        by_model: dict[str, int] = {}
        for e in entries:
            by_model[e.model_name] = by_model.get(e.model_name, 0) + 1
        return {
            "total_decisions": n,
            "deviation_breaches": n_breach,
            "hitl_required": n_hitl,
            "by_model": by_model,
            "path": str(self.path),
        }


__all__ = ["DecisionLog", "DecisionEntry"]
