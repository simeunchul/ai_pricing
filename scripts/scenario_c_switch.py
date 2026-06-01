"""Scenario C — 청산 + 변동성 ETF 진입.

순서:
  1) 현재 보유 종목 일괄 시장가 매도
  2) 가용 현금 확인
  3) 252710 KODEX 200 인버스 2X (50%) + 233740 KODEX 코스닥150 레버리지 (50%) 시장가 매수

사용 후 즉시 폐기 (one-shot 스크립트).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

# load env
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

NEW_INV2X = "252710"   # KODEX 200 인버스 2X
NEW_LEV2X = "233740"   # KODEX 코스닥150 레버리지

c = KISClient(KISConfig.from_env())
print(f"KIS env={c.cfg.env} dry_run={c.cfg.dry_run}")
print()

# ────────────────────────────────────── Step 1. 보유 조회
print("=" * 60)
print("Step 1: 보유 종목 조회")
print("=" * 60)
b = c.balance()
holdings = [h for h in (b.get("output1") or []) if int(h.get("hldg_qty", 0)) > 0]
o2 = b.get("output2", [{}])[0] if b.get("output2") else {}
print(f"  예수금: {int(o2.get('dnca_tot_amt',0)):,}원  총평가: {int(o2.get('tot_evlu_amt',0)):,}원")
for h in holdings:
    print(f"  {h.get('pdno')} qty={int(h.get('hldg_qty',0))} 평가={int(h.get('evlu_amt',0)):,}원")

# ────────────────────────────────────── Step 2. 청산
print()
print("=" * 60)
print("Step 2: 일괄 시장가 매도")
print("=" * 60)
sell_results = []
for h in holdings:
    sym = h.get("pdno")
    qty = int(h.get("hldg_qty", 0))
    if qty <= 0:
        continue
    print(f"  매도 {sym} qty={qty} (시장가)...", end=" ", flush=True)
    try:
        r = c.order(symbol=sym, qty=qty, side="sell", price=0, ord_div="01")
        rt = r.get("rt_cd", "?")
        msg = r.get("msg1", "")[:50]
        odno = r.get("output", {}).get("ODNO", "") if isinstance(r.get("output"), dict) else ""
        print(f"rt_cd={rt} ODNO={odno} {msg}")
        sell_results.append((sym, qty, rt, msg))
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {str(e)[:80]}")
        sell_results.append((sym, qty, "ERR", str(e)[:50]))
    time.sleep(1.2)  # quota safety

# ────────────────────────────────────── Step 3. 정산 대기 + 가용 현금
print()
print("=" * 60)
print("Step 3: 가용 현금 확인 (3초 대기)")
print("=" * 60)
time.sleep(3)
b2 = c.balance()
o2_new = b2.get("output2", [{}])[0] if b2.get("output2") else {}
holdings_after = [h for h in (b2.get("output1") or []) if int(h.get("hldg_qty", 0)) > 0]

cash_raw = int(o2_new.get("dnca_tot_amt", 0))
total_eval = int(o2_new.get("tot_evlu_amt", 0))
invested = sum(int(h.get("evlu_amt", 0)) for h in holdings_after)
cash_real = total_eval - invested
print(f"  예수금 (raw):       {cash_raw:,}원")
print(f"  총평가:             {total_eval:,}원")
print(f"  잔여 보유 평가:      {invested:,}원")
print(f"  실 가용 현금:        {cash_real:,}원")
if holdings_after:
    print("  잔여 보유:")
    for h in holdings_after:
        print(f"    {h.get('pdno')} qty={int(h.get('hldg_qty',0))} 평가={int(h.get('evlu_amt',0)):,}원")

# 보수적: cash_real 또는 cash_raw 중 작은 값 사용 (overcommit 방지)
cash_avail = max(0, min(cash_real, cash_raw))
print(f"  → 신규 진입 사용 가능: {cash_avail:,}원")

# ────────────────────────────────────── Step 4. 신규 매수 시세 조회
print()
print("=" * 60)
print("Step 4: 신규 종목 시세")
print("=" * 60)
prices = {}
for sym in (NEW_INV2X, NEW_LEV2X):
    q = c.quote(sym)
    p = float(q.get("output", {}).get("stck_prpr", 0))
    prices[sym] = p
    print(f"  {sym} 현재가: {p:,.0f}원")
    time.sleep(1.0)

# ────────────────────────────────────── Step 5. 신규 매수
print()
print("=" * 60)
print("Step 5: 신규 시장가 매수 (각 50%)")
print("=" * 60)
half = cash_avail // 2
for sym in (NEW_INV2X, NEW_LEV2X):
    p = prices[sym]
    if p <= 0:
        print(f"  {sym} 시세 0 → 스킵")
        continue
    qty = int(half // p)
    if qty <= 0:
        print(f"  {sym} qty={qty} (현금 부족) → 스킵")
        continue
    name = "KODEX 200 인버스 2X" if sym == NEW_INV2X else "KODEX 코스닥150 레버리지"
    print(f"  매수 {sym} {name} qty={qty} (~{int(qty*p):,}원)...", end=" ", flush=True)
    try:
        r = c.order(symbol=sym, qty=qty, side="buy", price=0, ord_div="01")
        rt = r.get("rt_cd", "?")
        msg = r.get("msg1", "")[:50]
        odno = r.get("output", {}).get("ODNO", "") if isinstance(r.get("output"), dict) else ""
        print(f"rt_cd={rt} ODNO={odno} {msg}")
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {str(e)[:80]}")
    time.sleep(1.2)

# ────────────────────────────────────── Step 6. 최종 확인
print()
print("=" * 60)
print("Step 6: 최종 잔고")
print("=" * 60)
time.sleep(3)
b3 = c.balance()
o2f = b3.get("output2", [{}])[0] if b3.get("output2") else {}
holdings_final = [h for h in (b3.get("output1") or []) if int(h.get("hldg_qty", 0)) > 0]
print(f"  예수금:   {int(o2f.get('dnca_tot_amt',0)):,}원")
print(f"  총평가:   {int(o2f.get('tot_evlu_amt',0)):,}원")
print(f"  손익:     {int(o2f.get('asst_icdc_amt',0)):+,}원 ({float(o2f.get('asst_icdc_erng_rt',0)):+.4f}%)")
print()
for h in holdings_final:
    print(f"  {h.get('pdno')} qty={int(h.get('hldg_qty',0))} avg={float(h.get('pchs_avg_pric',0)):,.0f} now={float(h.get('prpr',0)):,.0f} 평가={int(h.get('evlu_amt',0)):,}원")

print()
print("✓ Scenario C 진입 완료. 모니터링은 dashboard 로 (다른 종목 코드 기준).")
