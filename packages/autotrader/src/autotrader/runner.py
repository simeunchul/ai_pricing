"""Main event loop for KIS ETF iNAV strategy.

Usage:
    # Dry run (mocks KIS API, uses simulated prices)
    python -m autotrader.runner --symbol 069500 --dry --seconds 60 --log data/runner_log.json

    # Paper trading (hits real KIS API but DRY_RUN=true blocks orders)
    KIS_APP_KEY=... KIS_APP_SECRET=... KIS_ACCOUNT=... KIS_ENV=vts \\
        python -m autotrader.runner --symbol 069500 --seconds 300

Live orders require explicit `KIS_DRY_RUN=false` in env AND `--i-understand-live`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from autotrader.broker.kis_client import KISClient, KISConfig
from autotrader.market.inav import InavEstimator, deviation
from autotrader.risk.limits import RiskLimits, RiskState, check
from autotrader.strategies.etf_inav_arb import EtfInavArbitrage, Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("runner")


# Default KODEX 200 proxy basket (abbreviated). Real impl should load weights from issuer file.
DEFAULT_BASKET = {
    "005930": 0.30,  # 삼성전자
    "000660": 0.10,  # SK하이닉스
    "207940": 0.05,  # 삼성바이오로직스
    "005380": 0.04,  # 현대차
    "035420": 0.04,  # NAVER
}
DEFAULT_CASH_WEIGHT = 1.0 - sum(DEFAULT_BASKET.values())


def mock_prices(seed: int, tick: int) -> tuple[float, dict[str, float]]:
    """For --dry mode without API access. Adds gaussian noise to deterministic prices."""
    rng = np.random.default_rng(seed + tick)
    prices = {
        "069500": 42000 + rng.normal(0, 30),
        "005930": 75000 + rng.normal(0, 100),
        "000660": 125000 + rng.normal(0, 200),
        "207940": 850000 + rng.normal(0, 1500),
        "005380": 240000 + rng.normal(0, 400),
        "035420": 180000 + rng.normal(0, 300),
    }
    return prices["069500"], {k: v for k, v in prices.items() if k != "069500"}


def run(
    symbol: str = "069500",
    seconds: int = 60,
    poll_interval: float = 2.0,
    log_path: str = "data/runner_log.json",
    basket: dict[str, float] = None,
    cash_weight: float = None,
    live_opt_in: bool = False,
    dry_only: bool = False,
):
    """Main loop. Safe by default."""
    basket = basket or DEFAULT_BASKET
    cash_weight = cash_weight if cash_weight is not None else DEFAULT_CASH_WEIGHT
    inav_est = InavEstimator(constituents=basket, cash_weight=cash_weight * 1.0)
    strategy = EtfInavArbitrage()
    limits = RiskLimits()
    state = RiskState(equity_open=10_000_000, equity_now=10_000_000, position_value=0)

    cfg = KISConfig.from_env()
    if not dry_only and cfg.app_key and cfg.app_secret:
        if cfg.env == "prod" and not cfg.dry_run and not live_opt_in:
            raise RuntimeError("Refusing live trading without --i-understand-live")
        client = KISClient(cfg)
        logger.info(f"KIS client configured env={cfg.env} dry_run={cfg.dry_run}")
    else:
        client = None
        logger.info("Running in pure simulation (no KIS credentials or --dry)")

    log: list[dict] = []
    t_start = time.time()
    tick = 0
    while time.time() - t_start < seconds:
        now = datetime.now()
        try:
            # 1. Fetch prices
            if client is not None:
                q = client.quote(symbol)
                etf_price = float(q.get("output", {}).get("stck_prpr", 0)) or 0.0
                constituents = {}
                for sym in basket:
                    sq = client.quote(sym)
                    constituents[sym] = float(sq.get("output", {}).get("stck_prpr", 0)) or 0.0
            else:
                etf_price, constituents = mock_prices(seed=42, tick=tick)

            inav = inav_est.estimate(constituents)
            dev = deviation(etf_price, inav)

            # 2. Risk check
            ok, reason = check(state, limits, now)
            signal = Signal.HOLD
            action_taken = None
            if ok:
                signal = strategy.decide(dev)
                if signal != Signal.HOLD:
                    action_taken = signal.value
                    if client is not None:
                        resp = client.order(symbol, qty=1, side=signal.value.lower())
                        logger.info(f"order response: {resp}")
                    strategy.apply(signal)
                    state.error_count = 0
            else:
                logger.warning(f"risk halt: {reason}")

            record = {
                "ts": now.isoformat(),
                "etf": etf_price,
                "inav": inav,
                "dev_bps": dev * 10_000,
                "signal": signal.value,
                "action": action_taken,
                "position": strategy.position,
                "risk_ok": ok,
                "risk_reason": reason,
            }
            log.append(record)
            logger.info(f"t={tick} ETF={etf_price:.1f} iNAV={inav:.1f} "
                        f"dev={dev*1e4:+.1f}bps sig={signal.value} pos={strategy.position}")
        except Exception as e:
            state.error_count += 1
            logger.error(f"tick error: {e}")

        tick += 1
        time.sleep(poll_interval)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(json.dumps(log, indent=2, ensure_ascii=False))
    logger.info(f"wrote {len(log)} log entries to {log_path}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="069500", help="ETF symbol (default KODEX 200)")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--log", default="data/runner_log.json")
    ap.add_argument("--dry", action="store_true",
                    help="Force full simulation (no KIS calls even if creds set)")
    ap.add_argument("--i-understand-live", action="store_true",
                    help="Required for env=prod + KIS_DRY_RUN=false")
    args = ap.parse_args()
    run(args.symbol, args.seconds, args.poll, args.log,
        live_opt_in=args.i_understand_live, dry_only=args.dry)


if __name__ == "__main__":
    main()
