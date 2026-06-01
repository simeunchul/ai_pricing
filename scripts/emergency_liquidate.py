"""KIS 모의투자 phantom margin debt 응급 청산.

KIS vts 가 가용현금을 체크 안 하고 매수를 받아주는 quirk 때문에
'총평가 < 보유평가' 인 over-committed 상태가 생길 수 있다.
이 스크립트는 보유종목 일부를 시장가 매도하여 가용현금 = 총평가 - 보유평가 가
다시 양수가 되게 만든다.

Usage:
  python scripts/emergency_liquidate.py --dry-run
  python scripts/emergency_liquidate.py
  python scripts/emergency_liquidate.py --target-cash-buffer 500000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# load .env
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from autotrader.broker.kis_client import KISClient, KISConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="계획만 출력 (실주문 X)")
    ap.add_argument("--target-cash-buffer", type=int, default=200_000,
                    help="청산 후 남길 가용현금 최소치 (default 20만원)")
    ap.add_argument("--max-sell-per-symbol", type=int, default=None,
                    help="종목당 매도 수량 cap (default 보유 전량)")
    args = ap.parse_args()

    c = KISClient(KISConfig.from_env())
    bal = c.balance()
    if bal.get("rt_cd") != "0":
        print(f"잔고 조회 실패: {bal.get('msg1')}")
        sys.exit(1)

    out2 = bal.get("output2", [{}])[0] if bal.get("output2") else {}
    out1 = [h for h in (bal.get("output1") or [])
            if int(h.get("hldg_qty", 0)) > 0]

    sum_eval = sum(int(h["evlu_amt"]) for h in out1)
    tot_eval = int(out2.get("tot_evlu_amt", 0))
    raw_cash = int(out2.get("dnca_tot_amt", 0))
    implied_cash = tot_eval - sum_eval

    print(f"=== 현재 상태 ===")
    print(f"  보유 평가합 (sum_eval) = {sum_eval:>15,}원")
    print(f"  총평가 (tot_evlu)       = {tot_eval:>15,}원")
    print(f"  KIS 예수금 (dnca)      = {raw_cash:>15,}원  (참고용 — vts에선 신뢰 X)")
    print(f"  ★ 실 가용현금          = {implied_cash:>+15,}원  (음수면 phantom debt)")
    print()

    need_to_free = args.target_cash_buffer - implied_cash
    if need_to_free <= 0:
        print(f"청산 불필요 (현금 buffer {args.target_cash_buffer:,}원 이상 확보)")
        return

    print(f"청산 필요액: {need_to_free:,}원")
    print()
    print(f"=== 보유 종목 ({len(out1)}) ===")
    for h in out1:
        print(f"  {h['pdno']} qty={int(h['hldg_qty']):>4} "
              f"cur={float(h['prpr']):>10,.0f} "
              f"eval={int(h['evlu_amt']):>13,}")

    # 평가금액 큰 순서로 청산 — 한 종목에서 다 빼는 게 슬리피지 분산
    plan = []
    remaining = need_to_free
    for h in sorted(out1, key=lambda x: -int(x["evlu_amt"])):
        if remaining <= 0:
            break
        sym = h["pdno"]
        held_qty = int(h["hldg_qty"])
        cur = float(h["prpr"])
        if cur <= 0:
            continue
        # 이 종목에서 빼야 할 금액 / 가격 → 수량 (올림)
        need_qty = int(remaining // cur) + (1 if remaining % cur > 0 else 0)
        sell_qty = min(held_qty, need_qty)
        if args.max_sell_per_symbol is not None:
            sell_qty = min(sell_qty, args.max_sell_per_symbol)
        if sell_qty > 0:
            est_proceeds = sell_qty * cur
            plan.append((sym, sell_qty, cur, est_proceeds))
            remaining -= est_proceeds

    print()
    print(f"=== 청산 계획 ({len(plan)} 건) ===")
    total_sell = 0
    for sym, qty, cur, val in plan:
        print(f"  SELL {sym} {qty}주 @ ~{cur:,.0f}원 ≈ {val:,.0f}원")
        total_sell += val
    print(f"  ─ 합계 ≈ {total_sell:,.0f}원")
    print(f"  → 예상 가용현금 변동: {implied_cash:+,} → {implied_cash + total_sell:+,.0f}원")

    if args.dry_run:
        print()
        print("[DRY RUN] 실주문 보내지 않음. --dry-run 빼면 실행")
        return

    if not plan:
        print("실행할 매도 없음")
        return

    print()
    print("실행 중...")
    for sym, qty, cur, _ in plan:
        try:
            resp = c.order(sym, qty=qty, side="sell", price=0, ord_div="01")
            rt = resp.get("rt_cd")
            msg = resp.get("msg1", "")[:60]
            mark = "✓" if rt == "0" else "✗"
            print(f"  {mark} SELL {sym} {qty}주 → rt_cd={rt} msg={msg}")
        except Exception as e:
            print(f"  ✗ SELL {sym} {qty}주 → exception: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(0.5)  # KIS API rate

    print()
    print("=== 청산 후 잔고 재조회 ===")
    time.sleep(2)
    bal2 = c.balance()
    if bal2.get("rt_cd") == "0":
        o2b = bal2.get("output2", [{}])[0] if bal2.get("output2") else {}
        o1b = [h for h in (bal2.get("output1") or [])
               if int(h.get("hldg_qty", 0)) > 0]
        sum2 = sum(int(h["evlu_amt"]) for h in o1b)
        tot2 = int(o2b.get("tot_evlu_amt", 0))
        cash2 = tot2 - sum2
        print(f"  보유 평가합 = {sum2:,}원")
        print(f"  총평가      = {tot2:,}원")
        print(f"  ★ 가용현금  = {cash2:+,}원  "
              f"(이전 {implied_cash:+,} → 변동 {cash2 - implied_cash:+,})")


if __name__ == "__main__":
    main()
