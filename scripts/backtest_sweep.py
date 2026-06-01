"""ETF iNAV 차익 봇 파라미터 sweep — 양의 기댓값 영역 탐색.

backtest_etf_arb.replay() 를 grid search 로 돌려 (enter_bps, exit_bps,
qty_per_step) 조합별 PnL/Sharpe/MaxDD 비교. 결과 CSV + HTML 표.

Usage:
  python scripts/backtest_sweep.py --log data/kis_trading_log_20260428.json
  python scripts/backtest_sweep.py --log ... --out data/sweep_20260428.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backtest_etf_arb import BacktestConfig, replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--out", default="data/sweep_results.csv")
    ap.add_argument("--fee-bps", type=float, default=1.5)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    args = ap.parse_args()

    log_path = Path(args.log)
    records = json.loads(log_path.read_text(encoding="utf-8"))
    print(f"loaded {len(records)} records from {log_path.name}")
    syms = ([s.strip() for s in args.symbols.split(",")] if args.symbols else None)

    # Sweep grid — round-trip 비용이 ~7bps 이므로 enter 를 비용보다 커야 의미.
    enter_grid = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]   # bps
    exit_grid_factor = [0.1, 0.2, 0.5]   # exit = enter × factor
    qty_grid = [5, 10, 20, 50]
    max_pos = 200

    print(f"\nSweep: {len(enter_grid)} enter × {len(exit_grid_factor)} exit_factor "
          f"× {len(qty_grid)} qty = {len(enter_grid)*len(exit_grid_factor)*len(qty_grid)} runs")
    print(f"비용 가정: fee={args.fee_bps}bps/side slip={args.slippage_bps}bps "
          f"→ round-trip ~{(args.fee_bps + args.slippage_bps)*2:.1f}bps")

    results = []
    for enter, ef, qty in product(enter_grid, exit_grid_factor, qty_grid):
        cfg = BacktestConfig(
            enter_bps=enter,
            exit_bps=enter * ef,
            qty_per_step=qty,
            max_position=max_pos,
            cash_buffer=0,
            initial_cash=10_000_000,
            fee_bps_per_side=args.fee_bps,
            slippage_bps=args.slippage_bps,
        )
        try:
            r = replay(records, cfg, syms)
        except Exception as e:
            print(f"  fail enter={enter} ef={ef} qty={qty}: {e}")
            continue
        results.append({
            "enter_bps": enter,
            "exit_bps": round(enter * ef, 3),
            "qty_per_step": qty,
            "max_position": max_pos,
            "pnl": round(r.pnl, 0),
            "pnl_pct": round(r.pnl_pct, 6),
            "n_trades": r.n_trades,
            "n_buy": r.n_buy,
            "n_sell": r.n_sell,
            "win_rate": round(r.win_rate, 4),
            "max_dd_pct": round(r.max_drawdown_pct, 6),
            "fee_total": round(r.total_fee, 0),
            "slip_total": round(r.total_slip_cost, 0),
            "sharpe_proxy": round(r.sharpe_proxy, 5),
            "avg_hold_sec": round(r.avg_holding_ticks, 1),
        })

    # CSV 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"\n→ saved {len(results)} rows to {out_path}")

    # 정렬해서 top 10 출력 (PnL 기준)
    results_sorted = sorted(results, key=lambda x: -x["pnl"])
    print(f"\n=== Top 10 by PnL ===")
    print(f"{'enter':>6} {'exit':>6} {'qty':>4} {'pnl':>12} {'pnl%':>8} "
          f"{'trades':>6} {'win%':>6} {'maxDD%':>7} {'sharpe':>8}")
    for r in results_sorted[:10]:
        print(f"{r['enter_bps']:>6} {r['exit_bps']:>6} {r['qty_per_step']:>4} "
              f"{r['pnl']:>+12,.0f} {r['pnl_pct']*100:>+7.3f}% "
              f"{r['n_trades']:>6} {r['win_rate']*100:>5.1f}% "
              f"{r['max_dd_pct']*100:>6.2f}% {r['sharpe_proxy']:>+8.4f}")

    # 양의 PnL 만
    positive = [r for r in results_sorted if r["pnl"] > 0]
    print(f"\n양의 PnL 조합: {len(positive)} / {len(results)}")


if __name__ == "__main__":
    main()
