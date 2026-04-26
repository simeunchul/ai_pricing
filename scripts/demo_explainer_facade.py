"""ExplainerFacade demo — 누적된 audit_log 23 entries 를 2-tier 로 enrich.

목적:
  1. 기존 entries (customer_layer=None) 가 facade 통과 시 자연어로 채워짐
  2. 5 templates × distinct (model, stage) 페어 적어도 1개씩 시연
  3. unknown (model, stage) → generic fallback 안전 작동
  4. Backward compat — 신규 entry 를 customer_layer 직접 지정해서 append 가능

Usage:
  python scripts/demo_explainer_facade.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "compliance" / "src"))

from compliance import (
    DecisionLog, DecisionEntry, ExplainerFacade, TwoTierExplanation,
)


SEP = "=" * 78


def section(title: str):
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def main():
    log_path = ROOT / "data" / "compliance" / "audit_log.jsonl"
    out_path = ROOT / "data" / "compliance" / "audit_log_enriched.jsonl"

    section("ExplainerFacade Demo — 2-tier customer/regulator 설명")
    print(f"  audit log:  {log_path}")
    print(f"  enriched:   {out_path}")

    # ------------------------------------------------------------------
    # 1. Load existing entries (23 entries 누적)
    # ------------------------------------------------------------------
    log = DecisionLog(str(log_path))
    entries = log.read_all()
    n = len(entries)
    print(f"\n  [load] {n} entries 누적")
    if n == 0:
        print("  (audit log 비어있음 — desk_day.py 또는 bench_lp_strategies.py 먼저 실행)")
        return

    # 모델/스테이지 분포
    pairs = Counter((e.model_name, e.stage) for e in entries)
    print(f"\n  (model, stage) distribution:")
    for (m, s), c in pairs.most_common():
        print(f"    {m:<35s} / {s:<25s}  × {c}")

    # ------------------------------------------------------------------
    # 2. Apply facade
    # ------------------------------------------------------------------
    facade = ExplainerFacade()
    print(f"\n  [facade] {len(facade.known_pairs())} known templates:")
    for m, s in facade.known_pairs():
        print(f"    · {m} / {s}")

    explanations = facade.explain_all(entries)

    # ------------------------------------------------------------------
    # 3. distinct (model, stage) 1개씩 보여주기
    # ------------------------------------------------------------------
    section("Distinct (model, stage) 1개씩 — 2-tier 출력 샘플")
    seen = set()
    samples: list[TwoTierExplanation] = []
    for exp in explanations:
        key = (exp.model_name, exp.stage)
        if key in seen:
            continue
        seen.add(key)
        samples.append(exp)

    for i, exp in enumerate(samples, 1):
        print(f"\n  [{i}] {exp.model_name} / {exp.stage}")
        print(f"      decision_id : {exp.decision_id}")
        print(f"      ┌─ customer 층 (자연어, 영업·고객용)")
        print(f"      │  {exp.customer_layer}")
        print(f"      └─ regulator 층 (수치 raw, 감사관·내부 통제용)")
        reg = exp.regulator_layer
        print(f"         deviation_pct={reg.get('deviation_pct')} "
              f"breach={reg.get('deviation_breach')} "
              f"hitl={reg.get('hitl_required')}")
        out = reg.get("output")
        if isinstance(out, dict):
            for k, v in list(out.items())[:4]:
                print(f"         output.{k}={v}")
        else:
            print(f"         output={out}")

    # ------------------------------------------------------------------
    # 4. Generic fallback 시연 — 가짜 unknown entry
    # ------------------------------------------------------------------
    section("Generic fallback — unknown (model, stage)")
    fake = DecisionEntry(
        timestamp="2026-04-26T19:00:00",
        model_name="UnknownModel_v9",
        stage="experimental_stage",
        inputs={"x": 1.23},
        output={"y": 4.56},
        reference_value=4.20,
        deviation_pct=8.6,
        deviation_threshold_pct=5.0,
        deviation_breach=True,
        hitl_required=True,
    )
    fake_exp = facade.explain(fake)
    print(f"\n  unknown model 도 안전:")
    print(f"  customer: {fake_exp.customer_layer}")

    # ------------------------------------------------------------------
    # 5. Backward compat — 신규 entry 를 customer_layer 직접 지정해 append
    # ------------------------------------------------------------------
    section("Backward compat — append 시 customer_layer 직접 지정")
    custom_msg = "[demo] 신규 entry — customer_layer 를 직접 작성해서 audit_log 에 들어감"
    new_entry = log.append(
        model_name="DemoModel",
        stage="demo_stage",
        inputs={"foo": "bar"},
        output={"baz": 1.0},
        customer_layer=custom_msg,
    )
    print(f"\n  ✓ append OK — customer_layer 필드 작성됨:")
    print(f"    {new_entry.customer_layer}")

    # facade 가 explicit customer_layer 를 그대로 신뢰하는지
    re_exp = facade.explain(new_entry)
    assert re_exp.customer_layer == custom_msg, "facade 가 explicit layer 무시"
    print(f"  ✓ facade 가 explicit customer_layer 그대로 통과")

    # ------------------------------------------------------------------
    # 6. enriched JSONL 저장
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for exp in explanations:
            f.write(json.dumps(exp.to_dict(), ensure_ascii=False, default=str) + "\n")
    print()
    print(SEP)
    print(f"  → enriched 저장: {out_path}")
    print(f"    {len(explanations)} entries (각 entry = customer 층 + regulator 층)")
    print(SEP)


if __name__ == "__main__":
    main()
