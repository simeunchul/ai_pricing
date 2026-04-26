"""편차 모니터링 (Deviation Monitor) — AI 기본법 대응 #2.

기준 모델 (BSM closed-form, Heston semi-analytic) 과 AI 모델 (NN Pricer, Buehler)
의 출력 편차를 측정. threshold 초과 시 breach flag → audit_log + HITL escalation.

Why: AI 모델이 학습 도메인을 벗어났거나 OOD 입력을 받았을 때 silent failure 방지.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviationResult:
    label: str
    model_value: float
    reference_value: float
    deviation_abs: float
    deviation_pct: float
    threshold_pct: float
    breach: bool
    severity: str            # "ok" / "warning" / "high"


class DeviationMonitor:
    """기준 모델 대비 편차 측정 + 등급화.

    severity 등급:
      - 'ok'      : |deviation| ≤ threshold
      - 'warning' : threshold < |deviation| ≤ 2 × threshold
      - 'high'    : |deviation| > 2 × threshold (HITL escalation)

    Usage:
        mon = DeviationMonitor()
        res = mon.check(label="SPX_call_NN_vs_BSM",
                        model_value=1.9773, reference_value=1.8444,
                        threshold_pct=5.0)
        if res.severity == "high":
            ...
    """

    def __init__(self, default_threshold_pct: float = 5.0):
        self.default_threshold_pct = default_threshold_pct

    def check(self, label: str, model_value: float, reference_value: float,
              threshold_pct: float | None = None) -> DeviationResult:
        thr = threshold_pct if threshold_pct is not None else self.default_threshold_pct
        if abs(reference_value) < 1e-9:
            dev_abs = abs(model_value - reference_value)
            dev_pct = float("inf") if dev_abs > 1e-9 else 0.0
        else:
            dev_abs = abs(model_value - reference_value)
            dev_pct = dev_abs / abs(reference_value) * 100.0

        breach = dev_pct > thr
        if dev_pct <= thr:
            severity = "ok"
        elif dev_pct <= 2 * thr:
            severity = "warning"
        else:
            severity = "high"

        return DeviationResult(
            label=label,
            model_value=float(model_value),
            reference_value=float(reference_value),
            deviation_abs=round(dev_abs, 6),
            deviation_pct=round(dev_pct, 4),
            threshold_pct=float(thr),
            breach=breach,
            severity=severity,
        )


__all__ = ["DeviationMonitor", "DeviationResult"]
