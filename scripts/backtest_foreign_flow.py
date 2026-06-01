"""외국인 순매수 추종 전략 backtest CLI.

Usage:
  python scripts/backtest_foreign_flow.py
  python scripts/backtest_foreign_flow.py --tickers 005930,000660,035420
  python scripts/backtest_foreign_flow.py --enter-threshold 0.03 --exit-days 1

Output:
  data/backtest_foreign_flow_<TS>.parquet      (per-trade)
  data/backtest_foreign_flow_<TS>_summary.json (집계 통계 + buy-hold baseline)
  Console: 종목별 분해 + 매수후보유 대비 초과수익
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

from autotrader.backtest.foreign_flow import (
    ForeignFlowBacktestConfig,
    run_foreign_flow_backtest,
    buy_and_hold_baseline,
    summarize_trades,
)


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 외국인 매매 동향이 시장에 큰 영향을 주는 대형주 5종 (default)
DEFAULT_TICKERS = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035420",  # NAVER
    "005380",  # 현대차
    "051910",  # LG화학
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                    help="콤마 구분 종목 코드 (default: 5 대형주)")
    ap.add_argument("--enter-threshold", type=float, default=0.05,
                    help="외국인 순매수 / 거래량 비율 임계값 (default 0.05)")
    ap.add_argument("--exit-days", type=int, default=2,
                    help="보유 일수 만기 (default 2)")
    ap.add_argument("--qty-per-signal", type=int, default=10)
    ap.add_argument("--max-position", type=int, default=50)
    ap.add_argument("--cost-bps", type=float, default=25.0,
                    help="왕복 거래비용 bps (default 25)")
    ap.add_argument("--max-pages", type=int, default=60,
                    help="Naver 페이지네이션 깊이 (1페이지 ≈ 10영업일)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    cfg = ForeignFlowBacktestConfig(
        symbols=tickers,
        enter_threshold=args.enter_threshold,
        exit_after_days=args.exit_days,
        qty_per_signal=args.qty_per_signal,
        max_position_per_symbol=args.max_position,
        cost_bps=args.cost_bps,
        max_pages=args.max_pages,
        use_cache=not args.no_cache,
        start=args.start,
        end=args.end,
    )

    print("=== Foreign Flow Backtest ===")
    print(f"tickers          : {len(tickers)} ({', '.join(tickers)})")
    print(f"enter threshold  : {args.enter_threshold:.3f}  (외국인 순매수 / 거래량)")
    print(f"exit after days  : {args.exit_days}")
    print(f"qty per signal   : {args.qty_per_signal}")
    print(f"max position     : {args.max_position}")
    print(f"cost (round-trip): {args.cost_bps} bps")
    print(f"history depth    : {args.max_pages} pages (~{args.max_pages*10} trading days)")
    print(f"date range       : {args.start or 'auto'} ~ {args.end or 'auto'}")
    print()

    print("[1/3] 데이터 로드 + 신호 생성...")
    trades = run_foreign_flow_backtest(cfg)
    if trades.empty:
        print("[!] No trades produced. Check tickers / thresholds / network.")
        return

    print(f"  → {len(trades)} trades 생성")

    print("[2/3] 매수후보유 baseline 계산...")
    baseline = buy_and_hold_baseline(cfg)
    print(f"  → {len(baseline)} symbols baseline")

    print("[3/3] 통계 집계...")
    summary = summarize_trades(trades, baseline=baseline)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"backtest_foreign_flow_{ts}.parquet"
    trades.to_parquet(parquet_path)
    summary_path = out_dir / f"backtest_foreign_flow_{ts}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"[saved] per-trade : {parquet_path}")
    print(f"[saved] summary   : {summary_path}")
    print()

    print("=== 전략 성과 ===")
    print(f"  N trades       : {summary['n_trades']:,}")
    print(f"  Hit rate       : {summary['hit_rate']*100:.1f}%")
    print(f"  Mean P&L       : {summary['mean_return']*100:+.3f}%")
    print(f"  Std P&L        : {summary['std_return']*100:.3f}%")
    print(f"  Sharpe / trade : {summary['sharpe_per_trade']:.3f}")
    print(f"  Sharpe ann.    : {summary['sharpe_annualized']:.3f}")
    print(f"  Total ret sum  : {summary['total_return_sum']*100:+.2f}%")
    print(f"  Best/Worst     : {summary['best_trade']*100:+.2f}% / "
          f"{summary['worst_trade']*100:+.2f}%")
    print(f"  Avg hold       : {summary['avg_hold_days']:.1f} days")
    print()

    print("=== 종목별 분해 ===")
    print(f"  {'symbol':<8} {'N':>4}  {'hit':>6}  {'mean':>8}  {'total':>8}  {'b&h':>8}  {'excess':>8}")
    for r in summary["by_symbol"]:
        sym = r["symbol"]
        bh = baseline.get(sym, 0)
        exc = r["total"] - bh
        print(f"  {sym:<8} {r['n']:>4}  {r['hit']*100:>5.1f}%  "
              f"{r['mean']*100:>+7.3f}%  {r['total']*100:>+7.2f}%  "
              f"{bh*100:>+7.2f}%  {exc*100:>+7.2f}%")
    print()

    print("=== vs 매수후보유 (전체 평균) ===")
    print(f"  전략 종목별 누적 평균 : {summary['total_return_sum']/len(baseline)*100:+.2f}%")
    print(f"  매수후보유 평균       : {summary['buyhold_mean_return']*100:+.2f}%")
    print(f"  초과수익 (excess)     : {summary['excess_vs_buyhold_mean']*100:+.2f}%")
    print()

    if summary["excess_vs_buyhold_mean"] > 0.02:
        print("✓ 가설 지지: 외국인 추종이 매수후보유보다 유의미하게 우월")
    elif summary["excess_vs_buyhold_mean"] > -0.02:
        print("◯ 가설 미정: 매수후보유와 유사 — 통계적 유의성 추가 검증 필요")
    else:
        print("✗ 가설 기각: 매수후보유가 더 좋음 — 단순 추종은 알파 없음")


if __name__ == "__main__":
    main()
