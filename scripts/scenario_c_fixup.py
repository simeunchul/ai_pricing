"""Scenario C fixup — 잔여 청산 + 인버스 1X 대체 진입.

문제 1: KOSDAQ 200 인버스 2X (252710) 모의투자 매매불가
  → KODEX 200 인버스 (114800) 또는 KODEX 200선물인버스 (252670) 로 대체
문제 2: 091160 39주 부분체결 잔여
  → 재매도

Try order:
  1) 091160 39주 매도
  2) 후보들 quote 시도 → 첫 매매가능 종목 매수
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from autotrader.broker.kis_client import KISClient, KISConfig

c = KISClient(KISConfig.from_env())
print(f"KIS env={c.cfg.env} dry_run={c.cfg.dry_run}")

# 인버스 ETF 후보 (vts 매매가능 우선순위 시도)
INVERSE_CANDIDATES = [
    ("114800", "KODEX 인버스"),                  # 1x KOSPI 200 인버스
    ("252670", "KODEX 200선물인버스"),           # 1x 선물인버스
    ("251340", "KODEX 코스닥150 선물인버스"),    # 1x 코스닥 인버스
]

# Step 1. 091160 잔여 매도
print()
print("=" * 60)
print("Step 1: 091160 잔여 매도")
print("=" * 60)
b = c.balance()
holdings = [h for h in (b.get("output1") or []) if int(h.get("hldg_qty", 0)) > 0]
qty_091160 = 0
for h in holdings:
    if h.get("pdno") == "091160":
        qty_091160 = int(h.get("hldg_qty", 0))
        break
print(f"  현재 091160 보유: {qty_091160}주")
if qty_091160 > 0:
    print(f"  매도 091160 qty={qty_091160}...", end=" ", flush=True)
    r = c.order(symbol="091160", qty=qty_091160, side="sell", price=0, ord_div="01")
    print(f"rt_cd={r.get('rt_cd')} {r.get('msg1','')[:50]}")
    time.sleep(2)

# Step 2. 인버스 후보 매매가능 검사
print()
print("=" * 60)
print("Step 2: 인버스 후보 매매가능 검사")
print("=" * 60)
chosen_inverse = None
for sym, name in INVERSE_CANDIDATES:
    try:
        q = c.quote(sym)
        rt = q.get("rt_cd")
        p = float(q.get("output", {}).get("stck_prpr", 0))
        if rt == "0" and p > 0:
            # 1주 dry-test 매수로 매매가능 확인 — 실제로는 매수 시도해야 알 수 있음
            print(f"  {sym} {name}: 시세 {p:,.0f}원 — 후보")
            chosen_inverse = (sym, name, p)
            break
        else:
            print(f"  {sym} {name}: 시세 조회 실패 rt_cd={rt}")
    except Exception as e:
        print(f"  {sym} {name}: ERROR {str(e)[:60]}")
    time.sleep(1.0)

if not chosen_inverse:
    print("  → 인버스 후보 모두 실패. 현금 보유.")
    sys.exit(1)

# Step 3. 가용 현금 + 매수
print()
print("=" * 60)
print("Step 3: 인버스 매수")
print("=" * 60)
time.sleep(3)
b2 = c.balance()
o2 = b2.get("output2", [{}])[0] if b2.get("output2") else {}
holdings2 = [h for h in (b2.get("output1") or []) if int(h.get("hldg_qty", 0)) > 0]
total_eval = int(o2.get("tot_evlu_amt", 0))
invested = sum(int(h.get("evlu_amt", 0)) for h in holdings2)
cash_real = total_eval - invested
cash_raw = int(o2.get("dnca_tot_amt", 0))
cash_avail = max(0, min(cash_real, cash_raw))
print(f"  실 가용 현금: {cash_avail:,}원 (raw={cash_raw:,}, total={total_eval:,}, invested={invested:,})")

sym, name, p = chosen_inverse
qty = int(cash_avail // p)
if qty <= 0:
    print(f"  qty={qty} → 매수 불가")
    sys.exit(1)
print(f"  매수 {sym} {name} qty={qty} (~{int(qty*p):,}원)...", end=" ", flush=True)
r = c.order(symbol=sym, qty=qty, side="buy", price=0, ord_div="01")
print(f"rt_cd={r.get('rt_cd')} {r.get('msg1','')[:50]}")

# Step 4. 최종 확인
print()
print("=" * 60)
print("Step 4: 최종 잔고")
print("=" * 60)
time.sleep(3)
b3 = c.balance()
o2f = b3.get("output2", [{}])[0] if b3.get("output2") else {}
holdings3 = [h for h in (b3.get("output1") or []) if int(h.get("hldg_qty", 0)) > 0]
print(f"  예수금: {int(o2f.get('dnca_tot_amt',0)):,}원  총평가: {int(o2f.get('tot_evlu_amt',0)):,}원")
print(f"  손익: {int(o2f.get('asst_icdc_amt',0)):+,}원 ({float(o2f.get('asst_icdc_erng_rt',0)):+.4f}%)")
print()
for h in holdings3:
    sym = h.get("pdno")
    qty_h = int(h.get("hldg_qty", 0))
    avg = float(h.get("pchs_avg_pric", 0))
    now = float(h.get("prpr", 0))
    eval_amt = int(h.get("evlu_amt", 0))
    pnl_amt = int(h.get("evlu_pfls_amt", 0))
    pnl_pct = float(h.get("evlu_pfls_rt", 0))
    print(f"  {sym} qty={qty_h:>4} avg={avg:>10,.0f} now={now:>10,.0f} 평가={eval_amt:>13,}원 손익={pnl_amt:+,} ({pnl_pct:+.2f}%)")
