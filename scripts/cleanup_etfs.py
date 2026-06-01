"""ETF 일괄 청산 — 보유 중인 모든 ETF 시장가 매도.

사용 시나리오:
  ─ ETF arb 봇 정지 후 기존 ETF positions 정리
  ─ 장중 (09:00~15:30 KST) 에만 동작 — 그 외엔 거절됨

Usage:
  python scripts/cleanup_etfs.py              # 즉시 시도
  python scripts/cleanup_etfs.py --dry-run    # 매도 시뮬만
"""

from __future__ import annotations

import argparse
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

env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from autotrader.broker.kis_client import KISClient, KISConfig


# 현재 보유 중인 ETF + 알려진 ETF code prefix
KNOWN_ETF_CODES = {
    "069500": "KODEX 200",
    "102110": "TIGER 200",
    "148020": "KBSTAR 200",
    "152100": "ARIRANG 200",
    "278530": "KODEX 200TR",
    "105190": "KINDEX 200",
    "229200": "KODEX 코스닥150",
    "091160": "KODEX 반도체",
    "305720": "KODEX 2차전지산업",
    "266420": "KODEX 헬스케어",
    "252670": "KODEX 200 인버스",
    "251340": "KODEX 코스닥150 선물인버스",
    "252710": "KODEX 200선물인버스2X",
    "232080": "TIGER 코스피200 레버리지",
    "226490": "KODEX 코스피",
    "114800": "KODEX 인버스",
    "233740": "KODEX 코스닥150 레버리지",
}


def _i(s):
    return int(str(s or "0").replace(",", "") or "0")


def is_etf_by_name(name: str) -> bool:
    """이름 기반 ETF 판별 — KODEX/TIGER/KBSTAR 등 prefix."""
    if not name:
        return False
    upper = name.upper()
    return any(p in upper for p in [
        "KODEX", "TIGER", "KBSTAR", "ARIRANG", "KINDEX",
        "HANARO", "ACE", "KOSEF", "PLUS", "RISE", "SOL ", "SMART",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = KISConfig.from_env()
    if not cfg.app_key:
        print("[ERROR] KIS_APP_KEY 미설정")
        sys.exit(1)
    client = KISClient(cfg)

    # Balance 조회
    b = None
    for attempt in range(3):
        try:
            b = client.balance()
            if b.get("rt_cd") == "0":
                break
        except Exception as e:
            print(f"  attempt {attempt+1} fail: {str(e)[:80]}")
            time.sleep(2)
    if not b or b.get("rt_cd") != "0":
        print("[ERROR] balance 조회 실패")
        sys.exit(1)

    # ETF holdings 추출
    etf_holdings = []
    for h in b.get("output1") or []:
        sym = h.get("pdno", "")
        qty = _i(h.get("hldg_qty"))
        if qty <= 0:
            continue
        name = (h.get("prdt_name") or "").strip()
        is_etf = sym in KNOWN_ETF_CODES or is_etf_by_name(name)
        if is_etf:
            etf_holdings.append({
                "sym": sym,
                "name": name or KNOWN_ETF_CODES.get(sym, sym),
                "qty": qty,
            })

    print(f"=== ETF 청산 대상 ({len(etf_holdings)}개) @ {datetime.now().isoformat()} ===")
    if not etf_holdings:
        print("  ETF 보유 없음 — 종료")
        return

    for h in etf_holdings:
        print(f"  {h['sym']} {h['name']:25s} {h['qty']:>4}주")

    if args.dry_run:
        print()
        print("--dry-run: 실제 매도 안 함")
        return

    print()
    print("=== 시장가 매도 ===")
    success = 0
    failures = []
    for h in etf_holdings:
        sym, qty = h["sym"], h["qty"]
        print(f"  {sym} {qty}주...", end=" ", flush=True)
        try:
            r = client.order(sym, qty=qty, side="sell", price=0, ord_div="01")
            rt = r.get("rt_cd", "?")
            msg = r.get("msg1", "")[:60]
            if rt == "0":
                print(f"✓ {msg}")
                success += 1
            else:
                print(f"✗ rt_cd={rt} {msg}")
                failures.append((sym, msg))
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:80]}")
            failures.append((sym, str(e)[:60]))
        time.sleep(1.5)

    print()
    print(f"=== 완료: 성공 {success} / 실패 {len(failures)} ===")
    if failures:
        for sym, err in failures:
            print(f"  ✗ {sym}: {err}")


if __name__ == "__main__":
    main()
