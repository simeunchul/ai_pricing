"""실시간 외국인+기관 신호 paper trading 모니터.

KIS API `foreign-institution-total` 을 1분마다 polling 해서 영웅문 화면 데이터를
받아 ForeignInstFlowFollow 전략의 매수/매도 신호를 산출한다.

본 스크립트의 역할:
  - 신호 모니터링 (실제 주문은 KIS dry_run=True 면 mock)
  - 매분 외국인+기관 가집계 → 종목별 ratio 계산 → strategy decide
  - 신호 발생 시 콘솔 로그 + 결정 (BUY/SELL/HOLD)

실제 주문 연결은 strategy decide 결과를 KIS client.order() 에 전달하면 가능.
본 스크립트는 일단 신호 모니터링까지만 (안전).

Usage:
  KIS_APP_KEY=... KIS_APP_SECRET=... KIS_ACCOUNT=... \
  python scripts/run_realtime_foreign_inst.py --interval 60 --max-iters 20
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
from autotrader.market.foreign_inst_realtime import (
    RealtimeFlowCache, compute_flow_ratios,
)
from autotrader.strategies.foreign_inst_flow import ForeignInstFlowFollow
from autotrader.strategies.etf_inav_arb import Signal


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_TICKERS = [
    "005930", "000660", "035420", "035720", "005380",
    "051910", "005490", "207940", "105560", "055550",
]


def fetch_volumes(client: KISClient, symbols: list[str]) -> dict[str, int]:
    """각 종목의 당일 누적 거래량 조회 (KIS quote endpoint)."""
    out = {}
    for sym in symbols:
        try:
            q = client.quote(sym)
            o = q.get("output", {})
            vol = int(str(o.get("acml_vol", "0")).replace(",", ""))
            out[sym] = vol
        except Exception as e:
            print(f"  [warn] quote {sym}: {e}")
            out[sym] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--interval", type=int, default=60,
                    help="polling interval in seconds (default 60)")
    ap.add_argument("--max-iters", type=int, default=10,
                    help="max iterations (default 10 = ~10 min)")
    ap.add_argument("--enter-threshold", type=float, default=0.05)
    ap.add_argument("--sizing-factor", type=float, default=200.0)
    ap.add_argument("--market-code", default="0001",
                    help="0000 전체 / 0001 KOSPI / 1001 KOSDAQ")
    ap.add_argument("--dry-only", action="store_true",
                    help="신호만 출력, 실제 주문 절대 안 함 (default 행동)")
    args = ap.parse_args()

    cfg = KISConfig.from_env()
    client = KISClient(cfg)
    print(f"=== Realtime Foreign+Institution Monitor ===")
    print(f"env={cfg.env}, dry_run={cfg.dry_run}")
    print(f"interval={args.interval}s, max_iters={args.max_iters}")
    print(f"market_code={args.market_code}")
    print()

    if not cfg.app_key:
        print("[ERROR] KIS_APP_KEY 환경변수 미설정 — 실행 중단")
        return

    # token sanity check
    try:
        client.token()
        print("[OK] KIS token acquired")
    except Exception as e:
        print(f"[ERROR] KIS token 발급 실패: {e}")
        return

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    universe_set = set(tickers)
    cache = RealtimeFlowCache(update_interval=args.interval)
    strat = ForeignInstFlowFollow(
        enter_threshold=args.enter_threshold,
        sizing_factor=args.sizing_factor,
        min_threshold=0.01,
    )

    print(f"watching {len(tickers)} symbols: {', '.join(tickers)}")
    print()

    for it in range(args.max_iters):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"--- iter {it+1}/{args.max_iters} @ {ts} ---")

        # 1. 외국인+기관 가집계 받기
        df_flow = cache.get(client, market_code=args.market_code)
        if df_flow.empty:
            print("  [warn] empty payload — skipping iter")
        else:
            # 2. universe 만 필터
            df_uni = df_flow[df_flow["symbol"].isin(universe_set)]
            if df_uni.empty:
                print(f"  [info] none of {len(tickers)} watch list in current ranking — universe wide call needed")
            else:
                # 3. 각 종목 거래량 조회 후 ratio 계산
                volumes = fetch_volumes(client, df_uni["symbol"].tolist())
                df_ratios = compute_flow_ratios(df_uni, daily_volumes=volumes)

                # 4. strategy decide
                for _, row in df_ratios.iterrows():
                    sym = row["symbol"]
                    flow = float(row["flow_ratio"])
                    inst = float(row["inst_ratio"])
                    sig, qty = strat.decide(sym, flow, inst)
                    pos = strat.positions.get(sym, 0)
                    if sig != Signal.HOLD:
                        print(f"  [{sym} {row['name']:<10}] "
                              f"flow={flow:+.4f}  inst={inst:+.4f}  "
                              f"→ {sig.value} {qty}주 (pos {pos})")
                    elif abs(flow) > 0.02 or abs(inst) > 0.02:
                        print(f"  [{sym} {row['name']:<10}] "
                              f"flow={flow:+.4f}  inst={inst:+.4f}  HOLD (pos {pos})")

                    # 신호 발생 시 strategy state 업데이트 (실제 주문은 별도)
                    if sig != Signal.HOLD and not args.dry_only:
                        # TODO: client.order() 호출 — 운영자 안전 검토 후 활성
                        strat.apply(sym, sig, qty)

        if it < args.max_iters - 1:
            time.sleep(args.interval)

    print()
    print("=== 종료 — 현재 strategy 포지션 ===")
    for sym, qty in strat.positions.items():
        print(f"  {sym}: {qty} 주")


if __name__ == "__main__":
    main()
