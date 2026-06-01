"""KIS 모의투자 계좌 보유 종목 전량 시장가 청산.

Phase 2 (dual paper trading 실주문) 활성 전 기존 보유 종목 정리용.

Usage:
  # 1. 보유 종목 확인 (dry run)
  python scripts/liquidate_kis_holdings.py --check

  # 2. 진짜 청산 (시장가 매도, 장중에만)
  python scripts/liquidate_kis_holdings.py --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

from autotrader.broker.kis_client import KISClient, KISConfig

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_dotenv(path: Path):
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def is_market_open(now=None):
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    return (h, m) >= (9, 0) and (h, m) <= (15, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="보유 종목만 확인 (dry run)")
    ap.add_argument("--execute", action="store_true", help="실제 시장가 매도 발사")
    ap.add_argument("--force-off-hours", action="store_true",
                    help="장외 시간이라도 강제 실행 (테스트용)")
    args = ap.parse_args()

    if not args.check and not args.execute:
        print("--check 또는 --execute 중 하나 선택")
        return

    load_dotenv(ROOT / ".env")
    cfg = KISConfig.from_env()
    client = KISClient(cfg)
    client.token()

    now = datetime.now()
    print(f"=== KIS 잔고 청산 스크립트 @ {now.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"env={cfg.env}, dry_run={cfg.dry_run}")
    print()

    if args.execute and not is_market_open(now) and not args.force_off_hours:
        print(f"❌ 장외 시간 — 매매 불가 (장중 09:00~15:30 또는 --force-off-hours)")
        return

    # 1. 잔고 조회
    bal = client.balance()
    output1 = bal.get("output1", [])
    if not output1:
        print("✅ 보유 종목 없음 — 청산할 게 없음")
        return

    print(f"=== 보유 종목 ({len(output1)}종) ===")
    targets = []
    for item in output1:
        code = item.get("pdno", "")
        name = item.get("prdt_name", "").strip()
        qty = int(item.get("hldg_qty", "0"))
        if qty <= 0:
            continue
        avg = item.get("pchs_avg_pric", "0")
        cur = item.get("prpr", "0")
        val = item.get("evlu_amt", "0")
        pnl = item.get("evlu_pfls_amt", "0")
        print(f"  {code} {name:<25} {qty}주 @ {avg} → {cur} (평가 {val}, P&L {pnl})")
        targets.append((code, name, qty))

    if args.check:
        print(f"\n--check 모드 — 청산 안 함. 실제 청산은 --execute 사용.")
        return

    # 2. 청산 실행
    print(f"\n=== 시장가 매도 시작 ({len(targets)}종) ===")
    failures = []
    for code, name, qty in targets:
        try:
            result = client.order(code, qty=qty, side="sell", price=0, ord_div="01")
            rt = result.get("rt_cd", "?")
            msg = result.get("msg1", "")
            ord_no = result.get("output", {}).get("ODNO", "")
            if rt == "0":
                print(f"  ✓ {code} {name} {qty}주 매도 주문 OK (ord_no={ord_no})")
            else:
                print(f"  ✗ {code} {name}: rt={rt} {msg}")
                failures.append((code, name, msg))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"  ✗ {code} {name}: {err[:80]}")
            failures.append((code, name, err[:80]))
        time.sleep(1.5)   # rate limit 회피

    print(f"\n[10초 대기 후 잔고 재조회...]")
    time.sleep(10)

    bal = client.balance()
    remaining = [it for it in bal.get("output1", []) if int(it.get("hldg_qty", "0")) > 0]
    if remaining:
        print(f"\n⚠️ 잔여 보유: {len(remaining)}종")
        for it in remaining:
            print(f"  {it.get('pdno')} {it.get('prdt_name')} {it.get('hldg_qty')}주")
        print("→ 미체결 가능, 다음 영업일 또는 KIS HTS 에서 직접 처리 권장")
    else:
        print(f"\n✅ 모든 종목 청산 완료")

    if failures:
        print(f"\n실패 {len(failures)}건:")
        for code, name, err in failures:
            print(f"  {code} {name}: {err}")


if __name__ == "__main__":
    main()
