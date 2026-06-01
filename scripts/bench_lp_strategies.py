"""Phase 2 — LP 전략 백테스트: Avellaneda-Stoikov vs iNAV 단방향.

데이터: yfinance KODEX 200 (069500.KS) 일봉 ~6개월. (KIS 분봉 API 인증 부재 fallback.)
       세션 1 일 = 1 timestep 으로 단순화. 실 분봉 백테스트는 KIS API 인증 후.

비교 전략:
  1) iNAV-only (기존 단방향 — 평균회귀):
     deviation = (price - iNAV) / iNAV
     iNAV proxy = price 의 5일 이동평균 (실 iNAV API 없음)
  2) Avellaneda-Stoikov (양방향 LP + 재고 페널티):
     매 step 양방향 호가 → 다음 step price 가 호가 안에 들어오면 체결

평가 지표:
  - Total PnL (realized + unrealized)
  - Sharpe-like ratio (mean / std)
  - Max drawdown
  - Inventory volatility (재고 회전율)
  - Number of fills

Compliance hook 통합:
  - 매 step PnL 변화를 audit log
  - Inventory 가 절대 limit 의 80% 초과 시 deviation breach + HITL escalate
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "compliance" / "src"))

from autotrader.strategies.avellaneda_stoikov import (
    ASConfig, ASState, compute_quote, fill_check, update_state, mark_to_market,
)
from autotrader.strategies.etf_inav_arb import EtfInavArbitrage, Signal
from compliance import DecisionLog, DeviationMonitor, HITLGate

AUDIT_LOG = DecisionLog(ROOT / "data" / "compliance" / "audit_log.jsonl")
MONITOR = DeviationMonitor(default_threshold_pct=10.0)
HITL = HITLGate(env="simulation")


def fetch_etf_prices(ticker: str = "069500.KS", period: str = "6mo",
                     cache_path: str = "data/lp/etf_prices.csv") -> np.ndarray:
    """yfinance 일봉 → close 가격 array."""
    import time as _time
    cache = ROOT / cache_path
    if cache.exists() and (_time.time() - cache.stat().st_mtime) < 86400:
        print(f"[etf] cache hit: {cache}")
        lines = cache.read_text(encoding="utf-8").splitlines()[1:]
        return np.array([float(line.split(",")[1]) for line in lines])

    import yfinance as yf
    print(f"[etf] fetching {ticker} period={period} ...")
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty for {ticker}")
    closes = df["Close"].values
    cache.parent.mkdir(parents=True, exist_ok=True)
    dates = [d.strftime("%Y-%m-%d") for d in df.index.date]
    cache.write_text("date,close\n" + "\n".join(f"{d},{c}" for d, c in zip(dates, closes)),
                     encoding="utf-8")
    print(f"[etf] {len(closes)} closes saved → {cache}")
    return closes


def backtest_avellaneda_stoikov(prices: np.ndarray, cfg: ASConfig) -> dict:
    """A-S 양방향 LP 백테스트.

    각 day t:
      mid_t = price[t]
      quote_t = compute_quote(state, mid_t, t_remaining)
      next mid = price[t+1]
      fill_check → 체결 여부 → update_state
    """
    state = ASState()
    n = len(prices)
    pnl_history = []
    inventory_history = []
    spread_history = []
    quotes_log = []
    n_breaches = 0
    n_hitl = 0

    log_returns = np.diff(np.log(prices))
    daily_sigma = float(log_returns.std())
    cfg.sigma = daily_sigma * 0.5

    for t in range(n - 1):
        t_remaining = 1.0 - t / max(n - 1, 1)
        mid = float(prices[t])
        quote = compute_quote(state, mid, t_remaining, cfg)
        quotes_log.append(quote)
        spread_history.append(quote.spread)

        mid_next = float(prices[t + 1])
        fill = fill_check(quote, mid_next, cfg)
        if fill != "none":
            update_state(state, fill, quote, qty=cfg.quote_size)

        # MTM
        mtm = mark_to_market(state, mid_next)
        pnl_history.append(mtm)
        inventory_history.append(state.inventory)

        # Compliance — inventory 가 80% 초과 시 deviation
        inv_ratio = abs(state.inventory) / cfg.inventory_limit
        if inv_ratio > 0.80:
            n_breaches += 1
            hitl = HITL.evaluate(
                label="lp_inventory_high",
                metric_value=inv_ratio, threshold=0.80,
                severity="high" if inv_ratio > 0.95 else "medium",
                reason=f"Avellaneda-Stoikov LP inventory {state.inventory} / {cfg.inventory_limit} ({inv_ratio*100:.0f}%) — 재고 한도 임박",
                suggested_action="Skew quote 강화 또는 manual flatten",
            )
            if hitl.triggered:
                n_hitl += 1
            if t % 10 == 0:
                AUDIT_LOG.append(
                    model_name="Avellaneda_Stoikov",
                    stage="lp_quote",
                    inputs={"mid": mid, "t_remaining": t_remaining,
                            "inventory": state.inventory, "gamma": cfg.gamma},
                    output={"bid": quote.bid, "ask": quote.ask, "spread": quote.spread},
                    reference_model="inventory_limit",
                    reference_value=cfg.inventory_limit,
                    deviation_pct=round(inv_ratio * 100, 2),
                    deviation_threshold_pct=80.0,
                    deviation_breach=True,
                    hitl_required=hitl.triggered,
                    hitl_reason=hitl.reason if hitl.triggered else None,
                    extra={"severity": hitl.severity},
                )

    # close out at last price
    final_mid = float(prices[-1])
    final_mtm = mark_to_market(state, final_mid)

    pnl_arr = np.array(pnl_history)
    inv_arr = np.array(inventory_history)
    rets = np.diff(pnl_arr) if len(pnl_arr) > 1 else np.array([0.0])
    sharpe_like = float(rets.mean() / (rets.std() + 1e-9)) if len(rets) > 0 else 0.0
    max_dd = float((pnl_arr - np.maximum.accumulate(pnl_arr)).min()) if len(pnl_arr) > 0 else 0.0

    return {
        "strategy": "Avellaneda-Stoikov",
        "n_days": n,
        "n_buys": state.n_buys,
        "n_sells": state.n_sells,
        "final_mtm": round(final_mtm, 4),
        "final_inventory": state.inventory,
        "sharpe_like": round(sharpe_like, 4),
        "max_drawdown": round(max_dd, 4),
        "mean_spread": round(float(np.mean(spread_history)), 6),
        "std_inventory": round(float(np.std(inv_arr)), 2),
        "compliance_breaches": n_breaches,
        "hitl_triggered": n_hitl,
        "pnl_history": [round(p, 4) for p in pnl_arr.tolist()],
        "inventory_history": [int(i) for i in inv_arr.tolist()],
    }


def backtest_inav_arbitrage(prices: np.ndarray, ma_window: int = 5,
                             enter_threshold: float = 0.005,
                             exit_threshold: float = 0.001) -> dict:
    """iNAV-proxy 단방향 평균회귀 백테스트.

    iNAV proxy = price 의 ma_window 일 이동평균.
    deviation = (price - iNAV) / iNAV.
    """
    strat = EtfInavArbitrage(
        enter_threshold=enter_threshold, exit_threshold=exit_threshold,
        max_position=20,
    )
    cash = 0.0
    pnl_history = []
    inventory_history = []
    last_action_price = None
    n_buys = 0
    n_sells = 0

    for t in range(ma_window, len(prices)):
        price = float(prices[t])
        inav = float(np.mean(prices[t - ma_window:t]))
        deviation = (price - inav) / inav

        signal = strat.decide(deviation)
        if signal == Signal.BUY:
            strat.apply(Signal.BUY)
            cash -= price
            n_buys += 1
            last_action_price = price
        elif signal == Signal.SELL:
            strat.apply(Signal.SELL)
            cash += price
            n_sells += 1
            last_action_price = price

        mtm = cash + strat.position * price
        pnl_history.append(mtm)
        inventory_history.append(strat.position)

    pnl_arr = np.array(pnl_history)
    inv_arr = np.array(inventory_history)
    rets = np.diff(pnl_arr) if len(pnl_arr) > 1 else np.array([0.0])
    sharpe_like = float(rets.mean() / (rets.std() + 1e-9)) if len(rets) > 0 else 0.0
    max_dd = float((pnl_arr - np.maximum.accumulate(pnl_arr)).min()) if len(pnl_arr) > 0 else 0.0
    final_mtm = pnl_history[-1] if pnl_history else 0.0

    return {
        "strategy": "iNAV-arbitrage (one-direction)",
        "n_days": len(prices),
        "n_buys": n_buys,
        "n_sells": n_sells,
        "final_mtm": round(final_mtm, 4),
        "final_inventory": int(strat.position),
        "sharpe_like": round(sharpe_like, 4),
        "max_drawdown": round(max_dd, 4),
        "ma_window": ma_window,
        "enter_threshold": enter_threshold,
        "pnl_history": [round(p, 4) for p in pnl_arr.tolist()],
        "inventory_history": [int(i) for i in inv_arr.tolist()],
    }


def main():
    print("=" * 78)
    print("  Phase 2 — LP 전략 백테스트: Avellaneda-Stoikov vs iNAV 단방향")
    print("=" * 78)

    print("\n[1] ETF prices fetch (069500.KS = KODEX 200) ...")
    prices = fetch_etf_prices("069500.KS", period="6mo")
    print(f"    n_days = {len(prices)}, range = {prices.min():.0f} ~ {prices.max():.0f}")

    print("\n[2] Avellaneda-Stoikov backtest — config sweep ...")
    configs = {
        "default": ASConfig(gamma=0.1, sigma=0.01, k=1.5,
                             inventory_limit=20, quote_size=1, tick_size=0.0005),
        "tight":   ASConfig(gamma=0.5, sigma=0.01, k=2.5,
                             inventory_limit=50, quote_size=1, tick_size=0.003),
        "wide":    ASConfig(gamma=1.5, sigma=0.01, k=4.0,
                             inventory_limit=50, quote_size=1, tick_size=0.005),
    }
    sweep_results = {}
    for name, c in configs.items():
        t0 = time.time()
        r = backtest_avellaneda_stoikov(prices, c)
        elapsed = time.time() - t0
        r["config_name"] = name
        r["config"] = c.__dict__
        sweep_results[name] = r
        print(f"  [{name:>8s}]  final={r['final_mtm']:+12.0f}  inv={r['final_inventory']:+3d}  "
              f"trades={r['n_buys']+r['n_sells']:3d}  sharpe={r['sharpe_like']:+.4f}  "
              f"max_dd={r['max_drawdown']:+10.0f}  spread={r['mean_spread']:.4f}  ({elapsed:.1f}s)")

    res_as = max(sweep_results.values(), key=lambda x: x["final_mtm"])
    print(f"\n    best config = '{res_as['config_name']}' with final MTM {res_as['final_mtm']:+.0f}")

    print("\n[3] iNAV-arbitrage backtest ...")
    res_inav = backtest_inav_arbitrage(prices, ma_window=5, enter_threshold=0.005,
                                         exit_threshold=0.001)
    print(f"    final MTM = {res_inav['final_mtm']:+.2f}  inventory final = {res_inav['final_inventory']}  "
          f"buys={res_inav['n_buys']} sells={res_inav['n_sells']}  "
          f"sharpe-like = {res_inav['sharpe_like']:+.4f}  max_dd = {res_inav['max_drawdown']:.2f}")

    print("\n[4] 비교 매트릭스")
    print(f"  {'Metric':<25s}  {'iNAV (one-way)':>18s}  {'A-S (two-way LP)':>18s}  {'A-S 우위':>10s}")
    metrics = [
        ("Final MTM",     res_inav["final_mtm"], res_as["final_mtm"]),
        ("Trades total",  res_inav["n_buys"]+res_inav["n_sells"],
                          res_as["n_buys"]+res_as["n_sells"]),
        ("Sharpe-like",   res_inav["sharpe_like"], res_as["sharpe_like"]),
        ("Max drawdown",  res_inav["max_drawdown"], res_as["max_drawdown"]),
    ]
    for name, a, b in metrics:
        adv = ""
        try:
            if name == "Max drawdown":
                adv = "✓" if abs(b) < abs(a) else ""
            else:
                adv = "✓" if b > a else ""
        except Exception:
            adv = ""
        print(f"  {name:<25s}  {a:>18}  {b:>18}  {adv:>10}")

    out_dir = ROOT / "data" / "lp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench_lp_{datetime.now().strftime('%Y-%m-%d')}.json"
    out_path.write_text(json.dumps({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "ticker": "069500.KS",
        "n_days": len(prices),
        "as_sweep": sweep_results,
        "results": {"avellaneda_stoikov_best": res_as, "inav_arbitrage": res_inav},
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n→ {out_path}")
    print(f"\nAudit log: {AUDIT_LOG.summary()}")


if __name__ == "__main__":
    main()
