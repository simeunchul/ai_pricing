"""ETF iNAV 차익 봇 backtest — 라이브 로그 재생.

라이브 봇이 KIS 시세로 쌓아둔 tick 로그를 그대로 재생해서, 같은 전략 코드를
파라미터만 바꿔가며 돌린다. dev_bps 는 그 시점의 진짜 시장 데이터에서
나온 값이므로 신뢰 가능 (look-ahead bias 없음).

Phase B1: skeleton — single-config replay + summary.
Phase B3: parameter sweep (enter/exit bps × qty × max_pos).

Usage:
  python scripts/backtest_etf_arb.py --log data/kis_trading_log_20260428.json
  python scripts/backtest_etf_arb.py --log ... --enter-bps 1.5 --exit-bps 0.3
  python scripts/backtest_etf_arb.py --log ... --symbols 069500,102110
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from autotrader.strategies.etf_inav_arb import EtfInavArbitrage, Signal


# Inverse ETF 매핑 (runner 와 동기화 유지)
INVERSE_MAP = {
    "069500": "252670",
    "102110": "252670",
    "152100": "252670",
    "278530": "252670",
    "105190": "252670",
}


@dataclass
class BacktestConfig:
    """파라미터 한 set. sweep 시 여러 set 만들어 실행."""
    enter_bps: float = 1.0
    exit_bps: float = 0.2
    qty_per_step: int = 5
    max_position: int = 50
    cash_buffer: int = 0          # 0 = phantom debt 금지 (현재 운영 정책)
    initial_cash: float = 10_000_000
    # 비용 — KIS 실거래 기준 (모의투자 0이지만 prod 전환 대비)
    fee_bps_per_side: float = 1.5  # 0.015% (실거래 ~)
    slippage_bps: float = 2.0      # 1-2 tick 가정
    etf_sale_tax_bps: float = 0.0  # ETF 거래세 면제


@dataclass
class BacktestResult:
    config: BacktestConfig
    pnl: float
    pnl_pct: float
    final_equity: float
    n_trades: int
    n_buy: int
    n_sell: int
    total_fee: float
    total_slip_cost: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_proxy: float          # naive: pnl_pct / std(returns) — sample only
    win_rate: float              # 매도 trade 중 양수 PnL 비율
    avg_holding_ticks: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


def _slippage_buy_price(p: float, bps: float) -> float:
    return p * (1 + bps / 10_000)


def _slippage_sell_price(p: float, bps: float) -> float:
    return p * (1 - bps / 10_000)


def replay(records: list[dict], cfg: BacktestConfig,
           symbols: list[str] | None = None) -> BacktestResult:
    """단일 config 로 로그 재생 → 결과 1개 반환."""
    if symbols:
        recs = [r for r in records if r.get("symbol") in symbols]
    else:
        recs = list(records)
    if not recs:
        raise ValueError("no records to replay (check --symbols)")

    # 시간 순 정렬
    recs = sorted(recs, key=lambda r: r["ts"])
    syms_present = sorted({r["symbol"] for r in recs})

    # 종목별 strategy
    strategies: dict[str, EtfInavArbitrage] = {
        sym: EtfInavArbitrage(
            enter_threshold=cfg.enter_bps / 10_000,
            exit_threshold=cfg.exit_bps / 10_000,
            qty_per_step=cfg.qty_per_step,
            max_position=cfg.max_position,
        )
        for sym in syms_present
    }

    cash = cfg.initial_cash
    last_etf_price: dict[str, float] = {}        # 종목별 최근 시세 (MtM 용)
    open_buy_basis: dict[tuple[str, bool], list[tuple[float, int]]] = defaultdict(list)
    # ↑ (symbol, use_inverse) → list of (entry_price, qty) FIFO. SELL 시 PnL 계산용.

    trades: list[dict] = []
    equity_curve: list[dict] = []
    total_fee = 0.0
    total_slip_cost = 0.0

    for r in recs:
        sym = r["symbol"]
        dev = float(r["dev_bps"]) / 10_000
        etf_price = float(r["etf_price"])
        if etf_price > 0:
            last_etf_price[sym] = etf_price
        ss = strategies[sym]

        sig, qty, use_inverse = ss.decide_aggressive(dev)

        if sig != Signal.HOLD and qty > 0 and etf_price > 0:
            # 인버스로 라우팅되면 매핑 확인 (실제 inverse 시세 별도 추적 안 하므로
            # ETF 시세를 proxy로 사용 — backtest 한정 단순화)
            if use_inverse and sym not in INVERSE_MAP:
                pass  # skip (no mapping)
            else:
                target_price = etf_price  # backtest 단순화: 동일 시세 가정
                if sig == Signal.BUY:
                    # cash gating
                    spendable = cash - cfg.cash_buffer
                    raw_cost = qty * target_price
                    if raw_cost > spendable:
                        max_qty = (int(spendable // target_price)
                                   if target_price > 0 else 0)
                        if max_qty <= 0:
                            qty = 0
                        else:
                            qty = max_qty
                    if qty > 0:
                        fill = _slippage_buy_price(target_price, cfg.slippage_bps)
                        fee = qty * fill * cfg.fee_bps_per_side / 10_000
                        slip_cost = qty * (fill - target_price)
                        cash -= qty * fill + fee
                        total_fee += fee
                        total_slip_cost += slip_cost
                        ss.apply_aggressive(sig, qty, use_inverse)
                        open_buy_basis[(sym, use_inverse)].append((fill, qty))
                        trades.append({
                            "ts": r["ts"], "symbol": sym,
                            "side": "BUY", "qty": qty, "price": fill,
                            "fee": fee, "slip": slip_cost,
                            "inverse": use_inverse, "dev_bps": r["dev_bps"],
                        })
                else:  # SELL
                    if qty > 0:
                        fill = _slippage_sell_price(target_price, cfg.slippage_bps)
                        fee = qty * fill * cfg.fee_bps_per_side / 10_000
                        tax = qty * fill * cfg.etf_sale_tax_bps / 10_000
                        slip_cost = qty * (target_price - fill)
                        cash += qty * fill - fee - tax
                        total_fee += fee + tax
                        total_slip_cost += slip_cost
                        # FIFO 청산하면서 trade-level PnL 계산
                        remain = qty
                        realized_pnl = 0.0
                        basis_list = open_buy_basis.get((sym, use_inverse), [])
                        while remain > 0 and basis_list:
                            entry_p, entry_q = basis_list[0]
                            take = min(remain, entry_q)
                            realized_pnl += take * (fill - entry_p)
                            if take == entry_q:
                                basis_list.pop(0)
                            else:
                                basis_list[0] = (entry_p, entry_q - take)
                            remain -= take
                        ss.apply_aggressive(sig, qty, use_inverse)
                        trades.append({
                            "ts": r["ts"], "symbol": sym,
                            "side": "SELL", "qty": qty, "price": fill,
                            "fee": fee + tax, "slip": slip_cost,
                            "realized_pnl": realized_pnl,
                            "inverse": use_inverse, "dev_bps": r["dev_bps"],
                        })

        # MtM equity (cash + 보유종목 시가 평가)
        eq = cash
        for s_, st in strategies.items():
            p = last_etf_price.get(s_, 0.0)
            eq += st.position * p
            eq += st.inverse_position * p   # backtest 단순화 (inverse 시세 미보유)
        equity_curve.append({"ts": r["ts"], "equity": eq, "cash": cash})

    # ─── stats
    final = equity_curve[-1]["equity"] if equity_curve else cfg.initial_cash
    pnl = final - cfg.initial_cash
    pnl_pct = pnl / cfg.initial_cash

    # max drawdown
    max_eq = cfg.initial_cash
    max_dd = 0.0
    for p in equity_curve:
        if p["equity"] > max_eq:
            max_eq = p["equity"]
        dd = max_eq - p["equity"]
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd / cfg.initial_cash if cfg.initial_cash > 0 else 0.0

    # sharpe proxy: per-tick return 의 평균 / std × √(ticks per day)
    if len(equity_curve) >= 2:
        rets = []
        prev = equity_curve[0]["equity"]
        for p in equity_curve[1:]:
            if prev > 0:
                rets.append((p["equity"] - prev) / prev)
            prev = p["equity"]
        if rets:
            mean_r = sum(rets) / len(rets)
            var_r = sum((x - mean_r)**2 for x in rets) / max(1, len(rets) - 1)
            std_r = var_r ** 0.5
            # 보수적: tick 마다의 sharpe (annualization 안 함, 데이터 짧음)
            sharpe = (mean_r / std_r) if std_r > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    sells = [t for t in trades if t["side"] == "SELL"]
    n_buy = sum(1 for t in trades if t["side"] == "BUY")
    n_sell = len(sells)
    wins = sum(1 for t in sells if t.get("realized_pnl", 0) > 0)
    win_rate = wins / n_sell if n_sell > 0 else 0.0

    # 평균 보유 tick — 단순 추정 (BUY-SELL 짝, 같은 sym/inverse)
    holding_durations = []
    open_buys = defaultdict(list)
    for t in trades:
        key = (t["symbol"], t["inverse"])
        if t["side"] == "BUY":
            open_buys[key].append(t["ts"])
        else:
            if open_buys[key]:
                # FIFO match — tick 간격 측정
                from datetime import datetime
                bts = open_buys[key].pop(0)
                try:
                    dt_b = datetime.fromisoformat(bts)
                    dt_s = datetime.fromisoformat(t["ts"])
                    holding_durations.append((dt_s - dt_b).total_seconds())
                except Exception:
                    pass
    avg_hold = (sum(holding_durations) / len(holding_durations)
                if holding_durations else 0.0)

    return BacktestResult(
        config=cfg,
        pnl=pnl,
        pnl_pct=pnl_pct,
        final_equity=final,
        n_trades=len(trades),
        n_buy=n_buy,
        n_sell=n_sell,
        total_fee=total_fee,
        total_slip_cost=total_slip_cost,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_proxy=sharpe,
        win_rate=win_rate,
        avg_holding_ticks=avg_hold,
        trades=trades,
        equity_curve=equity_curve,
    )


def print_summary(res: BacktestResult, label: str = ""):
    cfg = res.config
    print(f"\n=== Backtest result {label} ===")
    print(f"  config: enter={cfg.enter_bps}bps exit={cfg.exit_bps}bps "
          f"qty/step={cfg.qty_per_step} max_pos={cfg.max_position} "
          f"cash_buf={cfg.cash_buffer:,}")
    print(f"  비용 가정: fee={cfg.fee_bps_per_side}bps/side slip={cfg.slippage_bps}bps")
    print(f"  PnL              = {res.pnl:>+15,.0f}원 ({res.pnl_pct:+.4%})")
    print(f"  최종 equity      = {res.final_equity:>15,.0f}원")
    print(f"  매매 (BUY/SELL)  = {res.n_buy} / {res.n_sell} (총 {res.n_trades})")
    print(f"  총 수수료/세금   = {res.total_fee:>15,.0f}원")
    print(f"  총 슬리피지 비용 = {res.total_slip_cost:>15,.0f}원")
    print(f"  Max drawdown     = {res.max_drawdown:>15,.0f}원 ({res.max_drawdown_pct:.3%})")
    print(f"  Win rate (SELL)  = {res.win_rate:.1%}")
    print(f"  평균 보유 시간   = {res.avg_holding_ticks:.1f}초")
    print(f"  Sharpe proxy     = {res.sharpe_proxy:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="라이브 로그 JSON 경로")
    ap.add_argument("--symbols", default=None,
                    help="콤마 구분 종목 (생략 시 전체)")
    ap.add_argument("--enter-bps", type=float, default=1.0)
    ap.add_argument("--exit-bps", type=float, default=0.2)
    ap.add_argument("--qty-per-step", type=int, default=5)
    ap.add_argument("--max-position", type=int, default=50)
    ap.add_argument("--cash-buffer", type=int, default=0)
    ap.add_argument("--initial-cash", type=float, default=10_000_000)
    ap.add_argument("--fee-bps", type=float, default=1.5,
                    help="side 당 수수료 bps (default 1.5 = 0.015%, KIS 실거래)")
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"로그 파일 없음: {log_path}"); sys.exit(1)
    records = json.loads(log_path.read_text(encoding="utf-8"))
    print(f"loaded {len(records)} records from {log_path.name}")

    syms = ([s.strip() for s in args.symbols.split(",")]
            if args.symbols else None)
    if syms:
        print(f"필터: {syms}")

    cfg = BacktestConfig(
        enter_bps=args.enter_bps,
        exit_bps=args.exit_bps,
        qty_per_step=args.qty_per_step,
        max_position=args.max_position,
        cash_buffer=args.cash_buffer,
        initial_cash=args.initial_cash,
        fee_bps_per_side=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    res = replay(records, cfg, syms)
    print_summary(res, label=f"({log_path.name})")


if __name__ == "__main__":
    main()
