"""state ↔ KIS holdings sync 유틸리티.

사용 시나리오:
  - runner 가 매도 주문 실패 후 state 만 업데이트되어 drift 발생 (구 버그)
  - 수동으로 KIS 화면에서 매수/매도 (state 미반영)
  - Fresh start 후 state 비어있는데 KIS 에는 holdings 있음

동작:
  1. KIS balance API → 실제 holdings 가져옴
  2. state.positions 와 비교
  3. Orphan (KIS 에만 있음) → state 에 추가
  4. Ghost (state 에만 있음) → state 에서 제거
  5. 수량/평균가 mismatch → KIS 기준으로 정정

Usage:
  python scripts/sync_state_with_kis.py
  python scripts/sync_state_with_kis.py --dry-run   # 변경 없이 비교만
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# .env 로드
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from autotrader.broker.kis_client import KISClient, KISConfig


def _i(s):
    return int(str(s or "0").replace(",", "") or "0")


def _f(s):
    return float(str(s or "0").replace(",", "") or "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(ROOT / "data" / "dual_state.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="변경 없이 비교만 (실제 sync X)")
    args = ap.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"[ERROR] state 파일 없음: {state_path}")
        sys.exit(1)

    state = json.loads(state_path.read_text(encoding="utf-8"))

    # KIS holdings (retry 3회)
    cfg = KISConfig.from_env()
    if not cfg.app_key:
        print("[ERROR] KIS_APP_KEY 미설정")
        sys.exit(1)
    client = KISClient(cfg)

    b = None
    for attempt in range(3):
        try:
            b = client.balance()
            if b.get("rt_cd") == "0":
                break
        except Exception as e:
            print(f"  KIS attempt {attempt+1} fail: {str(e)[:100]}")
            time.sleep(2)
    if b is None or b.get("rt_cd") != "0":
        print(f"[ERROR] KIS balance 호출 실패")
        sys.exit(1)

    kis_holdings = {}
    for h in b.get("output1") or []:
        qty = _i(h.get("hldg_qty"))
        if qty <= 0:
            continue
        kis_holdings[h.get("pdno")] = {
            "qty": qty,
            "avg_price": _f(h.get("pchs_avg_pric")),
            "name": (h.get("prdt_name") or "").strip(),
        }

    state_positions = state.get("positions", {})

    print("=" * 60)
    print(f"State 위치: {state_path}")
    print(f"State positions: {len(state_positions)}개")
    print(f"KIS holdings:    {len(kis_holdings)}개")
    print("=" * 60)

    # Orphan: KIS 에 있지만 state 에 없음
    orphans = [s for s in kis_holdings if s not in state_positions]
    # Ghost: state 에 있지만 KIS 에 없음
    ghosts = [s for s in state_positions if s not in kis_holdings]
    # Mismatch: 둘 다 있지만 qty/avg 다름
    mismatches = []
    for sym in kis_holdings:
        if sym in state_positions:
            kis_q = kis_holdings[sym]["qty"]
            kis_a = kis_holdings[sym]["avg_price"]
            st_q = state_positions[sym].get("qty", 0)
            st_a = state_positions[sym].get("avg_entry", 0)
            if kis_q != st_q or abs(kis_a - st_a) > 1.0:
                mismatches.append((sym, kis_q, st_q, kis_a, st_a))

    print()
    if orphans:
        print(f"[ORPHAN] KIS 에 있지만 state 에 없음 — {len(orphans)}개:")
        for sym in orphans:
            h = kis_holdings[sym]
            print(f"  {sym} {h['name']:20s} qty={h['qty']:>4} avg={h['avg_price']:>10,.0f}")
    if ghosts:
        print(f"[GHOST] state 에 있지만 KIS 에 없음 — {len(ghosts)}개:")
        for sym in ghosts:
            print(f"  {sym}")
    if mismatches:
        print(f"[MISMATCH] qty/avg 불일치 — {len(mismatches)}개:")
        for sym, kq, sq, ka, sa in mismatches:
            print(f"  {sym}: KIS qty={kq} avg={ka:,.0f} | state qty={sq} avg={sa:,.0f}")
    if not orphans and not ghosts and not mismatches:
        print("[OK] state 와 KIS 완전 동기화. 변경 불필요.")
        return

    if args.dry_run:
        print()
        print("--dry-run 옵션: 실제 변경 안 함.")
        return

    # 적용
    today = datetime.now().strftime("%Y-%m-%d")
    new_positions = {}
    for sym, h in kis_holdings.items():
        existing = state_positions.get(sym, {})
        # 진입일 보존 (orphan 만 새로 표기)
        entry_date = existing.get("entry_date") if existing else f"?-recovered-{today}"
        new_positions[sym] = {
            "qty": h["qty"],
            "avg_entry": h["avg_price"],
            "entry_date": entry_date,
        }

    state["positions"] = new_positions
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    print()
    print("=" * 60)
    print(f"[APPLIED] state 동기화 완료")
    print(f"  Orphan {len(orphans)}개 추가")
    print(f"  Ghost {len(ghosts)}개 제거")
    print(f"  Mismatch {len(mismatches)}개 정정")
    print(f"  최종 positions: {len(new_positions)}개")
    print("=" * 60)


if __name__ == "__main__":
    main()
