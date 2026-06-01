"""실시간 시장 전체 Dual Confirmation 자동 스캔.

영웅문 사용 방식과 동일:
  매번 KIS 매수 상위 30 + 매도 상위 30 받아서 → 둘 다 임계값 통과한 종목 자동 발견.
  고정 universe 없음 — 시장 전체에서 그날 진짜 신호 종목만 출력.

매수 후보: foreign_ratio > +threshold AND inst_ratio > +threshold
매도 후보: foreign_ratio < -threshold AND inst_ratio < -threshold

Usage:
  python scripts/scan_realtime_dual_confirmation.py --threshold 0.05
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

import pandas as pd

from autotrader.broker.kis_client import KISClient, KISConfig
from autotrader.market.foreign_inst_realtime import (
    parse_foreign_institution_response, compute_flow_ratios,
)


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_dotenv(path: Path) -> None:
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def fetch_market_dual(
    client: KISClient, market_code: str, threshold: float, sleep_between: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """매수 상위 + 매도 상위 두 호출 → dual 통과 매수/매도 후보 분리 반환."""
    # 매수 상위 30
    payload_buy = client.foreign_institution_total(market_code=market_code, rank_sort="0")
    df_buy = parse_foreign_institution_response(payload_buy)
    df_buy = compute_flow_ratios(df_buy, daily_volumes=None)

    time.sleep(sleep_between)

    # 매도 상위 30
    payload_sell = client.foreign_institution_total(market_code=market_code, rank_sort="1")
    df_sell = parse_foreign_institution_response(payload_sell)
    df_sell = compute_flow_ratios(df_sell, daily_volumes=None)

    # dual confirmation
    buy_pass = df_buy[
        (df_buy["flow_ratio"] > threshold) & (df_buy["inst_ratio"] > threshold)
    ].copy()
    sell_pass = df_sell[
        (df_sell["flow_ratio"] < -threshold) & (df_sell["inst_ratio"] < -threshold)
    ].copy()

    return buy_pass, sell_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="dual confirmation 임계값 (default 5%)")
    ap.add_argument("--market", default="0001",
                    choices=["0001", "1001"],
                    help="0001 KOSPI / 1001 KOSDAQ")
    ap.add_argument("--also-relaxed", action="store_true",
                    help="추가로 threshold/2 통과 후보도 출력")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = KISConfig.from_env()
    if not cfg.app_key:
        print("[ERROR] KIS_APP_KEY 미설정")
        return

    client = KISClient(cfg)
    client.token()

    market_label = {"0001": "KOSPI", "1001": "KOSDAQ"}[args.market]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== Dual Confirmation 시장 스캔 ({market_label}) ===")
    print(f"시각      : {now}")
    print(f"임계값    : ±{args.threshold*100:.1f}% (외인+기관 둘 다)")
    print()

    buy_pass, sell_pass = fetch_market_dual(client, args.market, args.threshold)

    # === 매수 후보 ===
    print(f"[매수 신호] 외인+기관 둘 다 +{args.threshold*100:.1f}% 초과")
    if buy_pass.empty:
        print("  (없음)")
    else:
        print(f"  {'종목':<22} {'현재가':>10} {'외인%':>9} {'기관%':>9} {'합계qty':>13} {'등락':>8}")
        for _, r in buy_pass.iterrows():
            print(f"  {r['symbol']} {r['name']:<14} {r['price']:>10,} "
                  f"{r['flow_ratio']*100:>+8.2f}% {r['inst_ratio']*100:>+8.2f}% "
                  f"{r['sum_net_qty']:>13,} {r['change_pct']:>+7.2f}%")

    print()
    print(f"[매도 신호] 외인+기관 둘 다 -{args.threshold*100:.1f}% 미만 (보유 시 청산 후보)")
    if sell_pass.empty:
        print("  (없음)")
    else:
        print(f"  {'종목':<22} {'현재가':>10} {'외인%':>9} {'기관%':>9} {'합계qty':>13} {'등락':>8}")
        for _, r in sell_pass.iterrows():
            print(f"  {r['symbol']} {r['name']:<14} {r['price']:>10,} "
                  f"{r['flow_ratio']*100:>+8.2f}% {r['inst_ratio']*100:>+8.2f}% "
                  f"{r['sum_net_qty']:>13,} {r['change_pct']:>+7.2f}%")

    if args.also_relaxed:
        print()
        relaxed = args.threshold / 2
        print(f"[참고: 완화 임계값 ±{relaxed*100:.1f}% 통과 추가 후보]")
        time.sleep(1.5)
        b2, s2 = fetch_market_dual(client, args.market, relaxed)
        # 이미 strict 통과한 종목 제외
        strict_buy = set(buy_pass["symbol"]) if not buy_pass.empty else set()
        strict_sell = set(sell_pass["symbol"]) if not sell_pass.empty else set()
        relaxed_buy = b2[~b2["symbol"].isin(strict_buy)]
        relaxed_sell = s2[~s2["symbol"].isin(strict_sell)]
        if not relaxed_buy.empty:
            print(f"  추가 매수 후보 ({len(relaxed_buy)}종):")
            for _, r in relaxed_buy.iterrows():
                print(f"    {r['symbol']} {r['name']:<14} flow={r['flow_ratio']*100:+.2f}%  inst={r['inst_ratio']*100:+.2f}%")
        if not relaxed_sell.empty:
            print(f"  추가 매도 후보 ({len(relaxed_sell)}종):")
            for _, r in relaxed_sell.iterrows():
                print(f"    {r['symbol']} {r['name']:<14} flow={r['flow_ratio']*100:+.2f}%  inst={r['inst_ratio']*100:+.2f}%")

    print()
    print(f"=== 요약 ===")
    print(f"  매수 후보: {len(buy_pass)} 종")
    print(f"  매도 후보: {len(sell_pass)} 종")
    print(f"  → 영웅문 화면 동일. 그날그날 dual 통과한 종목만 진입.")


if __name__ == "__main__":
    main()
