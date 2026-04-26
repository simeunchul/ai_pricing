"""실주문 endpoint 1회 테스트 — 내일 09:15 정도 장중에 1회 실행 권장.

LOGIC:
  1. 잔고 조회 → 예수금 확인
  2. KODEX 200 시세 조회
  3. 1주 매수 (시장가)
  4. 잠시 대기 (체결 확인)
  5. 잔고 재조회 → 1주 보유 확인
  6. 1주 매도 (시장가)
  7. 잔고 재조회 → 0주 확인

WARNINGS:
  - 반드시 장중 (09:10~15:20) 에만 실행
  - 시장가 주문이라 미세한 슬리피지 발생 (보통 1~2 tick = 5~10원)
  - 모의투자 가상머니라 손실 무관
  - KIS_DRY_RUN 가 자동 false 로 강제됨 (이 스크립트 한정)

Usage (내일 09:15 KST):
  python scripts/test_real_order_oneshot.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))


def load_env():
    env = ROOT / ".env"
    if not env.exists(): return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def is_safe_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5: return False
    return dtime(9, 10) <= now.time() <= dtime(15, 20)


def fetch_balance_safe(client) -> tuple[int, int, int]:
    """Returns (cash_KRW, eval_KRW, KODEX200_qty). 모든 값 0 if 실패."""
    try:
        bal = client.balance()
        if bal.get("rt_cd") != "0":
            print(f"  ⚠ balance rt_cd={bal.get('rt_cd')} msg={bal.get('msg1','')}")
            return 0, 0, 0
        out2 = bal.get("output2", [{}])[0]
        cash = int(out2.get("dnca_tot_amt", 0))
        eval_amt = int(out2.get("tot_evlu_amt", 0))
        # 보유 KODEX 200
        qty = 0
        for h in bal.get("output1", []):
            if h.get("pdno") == "069500":
                qty = int(h.get("hldg_qty", 0))
                break
        return cash, eval_amt, qty
    except Exception as e:
        print(f"  ⚠ balance error: {type(e).__name__} {str(e)[:100]}")
        return 0, 0, 0


def main():
    load_env()
    # 안전 강제: 이 스크립트만 임시로 dry_run 해제 (env 자체는 건드리지 않음)
    os.environ["KIS_DRY_RUN"] = "false"

    from autotrader.broker.kis_client import KISClient, KISConfig
    cfg = KISConfig.from_env()
    if cfg.env != "vts":
        print(f"⛔ KIS_ENV={cfg.env}. 실계좌 의심 — 중단.")
        return
    if not cfg.account_no:
        print("⛔ KIS_ACCOUNT 미설정"); return

    now = datetime.now()
    if not is_safe_market_hours(now):
        print(f"⛔ 안전 매매 시간 (09:10~15:20) 외: 현재 {now.strftime('%H:%M')}")
        print(f"   내일 09:15 정도에 다시 실행하세요.")
        return

    client = KISClient(cfg)
    print("=" * 70)
    print("  KIS 실주문 1회 시험 — 1주 매수 → 매도")
    print("=" * 70)
    print(f"  계좌: {cfg.account_no}, 환경: {cfg.env}, dry_run: {cfg.dry_run}")
    print(f"  시간: {now.strftime('%H:%M:%S')}")

    # 1. 잔고
    print("\n[1] 시작 잔고")
    cash0, eval0, qty0 = fetch_balance_safe(client)
    print(f"    예수금: {cash0:,}원, 총평가: {eval0:,}원, KODEX200 보유: {qty0}주")

    # 2. 시세
    print("\n[2] KODEX 200 현재가")
    q = client.quote("069500")
    if q.get("rt_cd") != "0":
        print(f"  ⚠ {q.get('msg1','')}"); return
    price = int(q.get("output", {}).get("stck_prpr", 0))
    print(f"    {price:,}원")

    # 3. 매수
    print(f"\n[3] 1주 시장가 매수...")
    buy_resp = client.order("069500", qty=1, side="buy", price=0, ord_div="01")
    print(f"    rt_cd={buy_resp.get('rt_cd')} msg={buy_resp.get('msg1','')}")
    if buy_resp.get("rt_cd") != "0":
        print("  ⛔ 매수 실패 — 스크립트 종료. 잔고 영향 없음.")
        return
    odno = buy_resp.get("output", {}).get("ODNO", "?")
    print(f"    주문번호 ODNO={odno}")

    # 4. 체결 대기 + 잔고 재확인
    print("\n[4] 5초 대기 후 잔고 재조회...")
    time.sleep(5)
    cash1, eval1, qty1 = fetch_balance_safe(client)
    print(f"    예수금: {cash1:,}원 ({cash1-cash0:+,}), KODEX200 보유: {qty1}주 ({qty1-qty0:+})")
    if qty1 <= qty0:
        print("  ⚠ 매수 미체결 가능 (5초 안에 못 채워짐). 매도 시도 안 함.")
        return

    # 5. 매도
    print(f"\n[5] 1주 시장가 매도...")
    sell_resp = client.order("069500", qty=1, side="sell", price=0, ord_div="01")
    print(f"    rt_cd={sell_resp.get('rt_cd')} msg={sell_resp.get('msg1','')}")
    if sell_resp.get("rt_cd") != "0":
        print(f"  ⚠ 매도 실패. 1주 보유 상태 유지.")
        return

    # 6. 최종 잔고
    print("\n[6] 5초 대기 후 최종 잔고...")
    time.sleep(5)
    cash2, eval2, qty2 = fetch_balance_safe(client)
    print(f"    예수금: {cash2:,}원 ({cash2-cash0:+,}), KODEX200 보유: {qty2}주 ({qty2-qty0:+})")

    # 7. 종합
    print("\n" + "=" * 70)
    print("  결과")
    print("=" * 70)
    print(f"  순 슬리피지: {cash2-cash0:+,}원 (매수+매도 시장가 차이)")
    print(f"  최종 포지션: {qty2}주 (시작 {qty0}주, 차이 {qty2-qty0:+})")
    if qty2 == qty0:
        print(f"\n  ✓ 매수+매도 endpoint 모두 정상. runner 실행 안전.")
    else:
        print(f"\n  ⚠ 포지션 불균형 — 수동 정리 필요. KIS HTS 에서 확인.")


if __name__ == "__main__":
    main()
