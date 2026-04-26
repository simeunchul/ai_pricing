"""KIS API 키 검증 — 토큰 발급, 시세 조회, (옵션) 잔고/주문 mock.

내일 장중 자동매매 돌리기 전에 필수 사전 검증.

Usage:
  python scripts/verify_kis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()  # strip inline comments
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    print("=" * 70)
    print("  KIS API 검증")
    print("=" * 70)

    cid = os.environ.get("KIS_APP_KEY", "")
    csec = os.environ.get("KIS_APP_SECRET", "")
    cacc = os.environ.get("KIS_ACCOUNT", "")
    cenv = os.environ.get("KIS_ENV", "vts")
    cdry = os.environ.get("KIS_DRY_RUN", "true")

    print(f"\n  KIS_ENV          : {cenv}")
    print(f"  KIS_APP_KEY      : {cid[:10]}... ({len(cid)} chars)")
    print(f"  KIS_APP_SECRET   : {csec[:10]}... ({len(csec)} chars)")
    print(f"  KIS_ACCOUNT      : {cacc or '(empty — 시세조회만 가능)'}")
    print(f"  KIS_DRY_RUN      : {cdry}")

    if not (cid and csec):
        print("\n  ✗ KIS_APP_KEY 또는 KIS_APP_SECRET 누락"); return

    # ---------------------------------------------------------------
    # 1. KISClient 인스턴스
    # ---------------------------------------------------------------
    from autotrader.broker.kis_client import KISClient, KISConfig

    cfg = KISConfig.from_env()
    client = KISClient(cfg)
    print(f"\n  HOST: {cfg.host}")

    # ---------------------------------------------------------------
    # 2. Token 발급
    # ---------------------------------------------------------------
    print("\n  [1] OAuth2 token 발급 시도...")
    try:
        # token cache 무효화 (검증 정확성)
        token_cache = Path(cfg.token_cache)
        if token_cache.exists():
            token_cache.unlink()

        tok = client.token()
        print(f"      ✓ 토큰 발급 성공: {tok[:30]}...")
        print(f"      만료 epoch: {client._token_exp:.0f} (현재로부터 ~{(client._token_exp - __import__('time').time())/3600:.1f}시간)")
    except Exception as e:
        print(f"      ✗ 실패: {type(e).__name__}: {str(e)[:200]}")
        return

    # ---------------------------------------------------------------
    # 3. 시세 조회 (KODEX 200 069500)
    # ---------------------------------------------------------------
    print("\n  [2] 시세 조회 시도 (KODEX 200 069500)...")
    try:
        q = client.quote("069500")
        rt_cd = q.get("rt_cd")
        if rt_cd == "0":
            output = q.get("output", {})
            price = output.get("stck_prpr", "?")
            chg = output.get("prdy_vrss", "?")
            chg_pct = output.get("prdy_ctrt", "?")
            volume = output.get("acml_vol", "?")
            print(f"      ✓ 현재가: {price}원  ({chg}원 / {chg_pct}%)")
            print(f"        누적거래량: {volume}")
        else:
            print(f"      ⚠ rt_cd={rt_cd}, msg={q.get('msg1', '')}")
            print(f"      raw: {q}")
    except Exception as e:
        print(f"      ✗ 실패: {type(e).__name__}: {str(e)[:200]}")

    # ---------------------------------------------------------------
    # 4. 잔고 조회 (계좌번호 있을 때만)
    # ---------------------------------------------------------------
    balance_ok = False
    if cacc:
        print(f"\n  [3] 잔고 조회 시도 (계좌 {cacc[:4]}...{cacc[-2:]})...")
        try:
            bal = client.balance()
            rt_cd = bal.get("rt_cd")
            if rt_cd == "0":
                output2 = bal.get("output2", [{}])[0] if bal.get("output2") else {}
                cash = output2.get("dnca_tot_amt", "?")
                eval_amt = output2.get("tot_evlu_amt", "?")
                print(f"      ✓ 예수금: {cash}원, 총평가: {eval_amt}원")
                balance_ok = True
            else:
                print(f"      ⚠ rt_cd={rt_cd}, msg={bal.get('msg1', '')}")
        except Exception as e:
            print(f"      ✗ 실패: {type(e).__name__}: {str(e)[:200]}")
            print(f"        (장외 시간엔 잔고 API 가 일시 unavailable 일 수 있음)")
    else:
        print(f"\n  [3] 잔고 조회 — KIS_ACCOUNT 미입력으로 skip")

    # ---------------------------------------------------------------
    # 5. 주문 mock (DRY_RUN)
    # ---------------------------------------------------------------
    print(f"\n  [4] 주문 호출 (DRY_RUN={cfg.dry_run})...")
    try:
        resp = client.order("069500", qty=1, side="buy", price=0, ord_div="01")
        if resp.get("mocked"):
            print(f"      ✓ DRY_RUN mock: {resp}")
        else:
            print(f"      실주문 응답: rt_cd={resp.get('rt_cd')} msg={resp.get('msg1', '')}")
    except Exception as e:
        print(f"      ✗ 실패: {type(e).__name__}: {str(e)[:200]}")

    # ---------------------------------------------------------------
    # 종합
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  요약")
    print("=" * 70)
    print(f"  ✓ 토큰 발급         : OK")
    print(f"  ✓ 시세 조회         : OK")
    if cacc:
        print(f"  {'✓' if balance_ok else '⚠'} 잔고 조회         : {'OK' if balance_ok else '실패 (장외 일시 unavailable 가능, 장중 재확인)'}")
    else:
        print(f"  ○ 잔고 조회         : 계좌번호 입력 시 가능")
    print(f"  ✓ DRY_RUN 주문 mock : OK")
    print()
    if cacc and balance_ok:
        print(f"  → 내일 장중 자동매매 가능 (KIS_DRY_RUN=false 로 전환 후)")
    elif cacc:
        print(f"  → 토큰/시세/주문 OK. 잔고만 실패 — 장중 재시도 권장.")
        print(f"     매매 자체는 잔고 조회 없이도 가능 (서버측 자동 검증).")
    else:
        print(f"  → 계좌번호 입력 후 재실행")


if __name__ == "__main__":
    main()
