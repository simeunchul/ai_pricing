"""뉴스 신호 → T+5분 진입 / T+1시간 청산 backtest CLI.

Usage:
  python scripts/backtest_news_intraday.py
  python scripts/backtest_news_intraday.py --classifier finbert --hold-min 60
  python scripts/backtest_news_intraday.py --events earnings_beat,mna --hold-min 30

Output:
  data/backtest_news_intraday_<TS>.parquet      (per-trade)
  data/backtest_news_intraday_<TS>_summary.json (aggregate stats)
  Console: per-event/per-ticker breakdown
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from autotrader.backtest.news_intraday import (
    BacktestConfig, run_news_intraday_backtest, summarize_trades,
    POSITIVE_EVENTS,
)


# Console UTF-8 (Windows cp949 회피)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 뉴스 캐시에 있는 10 KOSPI 종목 (default)
DEFAULT_TICKERS = [
    "000660",  # SK하이닉스
    "005380",  # 현대차
    "005490",  # POSCO홀딩스
    "005930",  # 삼성전자
    "035420",  # NAVER
    "035720",  # 카카오
    "051910",  # LG화학
    "066570",  # LG전자
    "068270",  # 셀트리온
    "207940",  # 삼성바이오로직스
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=",".join(DEFAULT_TICKERS),
                    help="콤마 구분 종목 코드 (default: 10 KOSPI)")
    ap.add_argument("--classifier", choices=["rule", "finbert"], default="rule")
    ap.add_argument("--entry-delay-min", type=int, default=5)
    ap.add_argument("--hold-min", type=int, default=60)
    ap.add_argument("--cost-bps", type=float, default=27.0,
                    help="round-trip 거래비용 bps (default 27 = 0.27%)")
    ap.add_argument("--events", default=None,
                    help="콤마 구분 tradable events. default: earnings_beat,mna,rating_change")
    ap.add_argument("--news-dir", default=str(ROOT / "data" / "news_cache"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    tradable = (
        set(args.events.split(",")) if args.events else POSITIVE_EVENTS
    )

    cfg = BacktestConfig(
        tickers=tickers,
        news_dir=Path(args.news_dir),
        classifier=args.classifier,
        entry_delay_min=args.entry_delay_min,
        hold_min=args.hold_min,
        cost_bps=args.cost_bps,
        tradable_events=tradable,
    )

    print(f"=== News Intraday Backtest ===")
    print(f"tickers     : {len(tickers)} ({', '.join(tickers)})")
    print(f"classifier  : {args.classifier}")
    print(f"entry/exit  : T+{args.entry_delay_min}min / T+{args.entry_delay_min+args.hold_min}min")
    print(f"hold        : {args.hold_min}min")
    print(f"cost        : {args.cost_bps} bps round-trip")
    print(f"tradable    : {sorted(tradable)}")
    print(f"news cache  : {args.news_dir}")
    print()

    df = run_news_intraday_backtest(cfg)
    if df.empty:
        print("[!] No trades produced. Check news cache or tickers.")
        return

    # save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"backtest_news_intraday_{ts}.parquet"
    df.to_parquet(parquet_path)

    summary = summarize_trades(df, only_tradable=True)
    summary_all = summarize_trades(df, only_tradable=False)

    summary_path = out_dir / f"backtest_news_intraday_{ts}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"tradable_only": summary, "all_events": summary_all},
                  f, ensure_ascii=False, indent=2)

    # console output
    print(f"[saved] per-trade : {parquet_path}")
    print(f"[saved] summary   : {summary_path}")
    print()

    print(f"=== Tradable trades only (events: {sorted(tradable)}) ===")
    if summary["n"] == 0:
        print("  no tradable trades")
    else:
        print(f"  N trades       : {summary['n']:,}")
        print(f"  Hit rate       : {summary['hit_rate']*100:.1f}%")
        print(f"  Mean P&L       : {summary['mean_pnl']*100:+.3f}%")
        print(f"  Std P&L        : {summary['std_pnl']*100:.3f}%")
        print(f"  Sharpe / trade : {summary['sharpe_per_trade']:.3f}")
        print(f"  Sharpe ann.    : {summary['sharpe_annualized']:.3f}")
        print(f"  Total P&L      : {summary['total_pnl_sum']*100:+.2f}%")
        print(f"  Best/Worst     : {summary['best_trade']*100:+.2f}% / "
              f"{summary['worst_trade']*100:+.2f}%")
        print()
        print(f"  Per-event:")
        for e in summary["by_event"]:
            print(f"    {e['event']:18s}  N={e['n']:>4}  hit={e['hit']*100:5.1f}%  "
                  f"mean={e['mean']*100:+.3f}%")
        print()
        print(f"  Per-ticker (top/bottom 5):")
        bt = sorted(summary["by_ticker"], key=lambda r: -r["mean"])
        for e in bt[:5]:
            print(f"    {e['ticker']}  N={e['n']:>4}  hit={e['hit']*100:5.1f}%  "
                  f"mean={e['mean']*100:+.3f}%")
        if len(bt) > 5:
            print("    ...")
            for e in bt[-3:]:
                print(f"    {e['ticker']}  N={e['n']:>4}  hit={e['hit']*100:5.1f}%  "
                      f"mean={e['mean']*100:+.3f}%")

    print()
    print(f"=== All events (including non-tradable for diagnostics) ===")
    if summary_all["n"] > 0:
        print(f"  N (all)        : {summary_all['n']:,}")
        for e in summary_all["by_event"]:
            print(f"    {e['event']:18s}  N={e['n']:>4}  hit={e['hit']*100:5.1f}%  "
                  f"mean={e['mean']*100:+.3f}%")


if __name__ == "__main__":
    main()
