"""KIS 모의투자 실전 운영 스크립트 — 내일 장중 실행용.

특징:
  - 장 시작/종료 자동 정렬 (KST 09:00~15:30)
  - 장 시작/마감 10분 자동 차단 (risk guard)
  - 5초 간격 polling (KIS API quota 안전)
  - 모든 tick 을 JSON 로그 (audit trail)
  - 신호 발생 시 mock 또는 실주문 (KIS_DRY_RUN 따름)
  - 중간 SIGINT (Ctrl+C) 시 graceful shutdown + 로그 저장

Usage:
  # 1. 일반 운영 (장중 시간만, 자동 시작/종료)
  python scripts/run_kis_paper_trading.py

  # 2. 시간 무시하고 일정 시간만 (장외 테스트)
  python scripts/run_kis_paper_trading.py --duration 600 --ignore-hours

  # 3. 종일 운영 (장 끝나면 자동 종료)
  python scripts/run_kis_paper_trading.py --until-close

  # 환경변수: KIS_DRY_RUN=true 면 실주문 안 나감 (mock).
  # 실제 모의투자 가상머니 매매 하려면 .env 에서 KIS_DRY_RUN=false 로.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "ai_pricing" / "src"))

from autotrader.broker.kis_client import KISClient, KISConfig
from autotrader.market.inav import InavEstimator, deviation
from autotrader.risk.limits import RiskLimits, RiskState, check
from autotrader.strategies.etf_inav_arb import EtfInavArbitrage, Signal


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# KODEX 200 (069500) 구성종목 상위 + 가중치 (대략, 매월 발행사 발표).
# 정확한 weights 는 https://www.samsungfund.com 의 KODEX 200 PDF 참조.
KODEX200_BASKET_TOP10 = {
    "005930": 0.30,   # 삼성전자
    "000660": 0.07,   # SK하이닉스
    "207940": 0.04,   # 삼성바이오로직스
    "005380": 0.04,   # 현대차
    "035420": 0.03,   # NAVER
    "000270": 0.02,   # 기아
    "005490": 0.02,   # POSCO홀딩스
    "035720": 0.02,   # 카카오
    "051910": 0.02,   # LG화학
    "068270": 0.02,   # 셀트리온
}
CASH_WEIGHT = 1.0 - sum(KODEX200_BASKET_TOP10.values())  # 0.42 ≈ 미포함 종목 + cash


# ----- KST market hours --------------------------------------------------
KST_OPEN = dtime(9, 0)
KST_CLOSE = dtime(15, 30)
SAFE_OPEN = dtime(9, 10)     # 장시작 10분 후
SAFE_CLOSE = dtime(15, 20)   # 장마감 10분 전


def is_market_open(now: datetime) -> bool:
    if now.weekday() >= 5:    # 주말
        return False
    return KST_OPEN <= now.time() <= KST_CLOSE


def is_safe_to_trade(now: datetime) -> bool:
    if not is_market_open(now):
        return False
    return SAFE_OPEN <= now.time() <= SAFE_CLOSE


def seconds_until_market_open(now: datetime) -> float:
    """장이 닫혀있으면 다음 장 개장까지 초. 장중이면 0."""
    if is_market_open(now):
        return 0.0
    today_open = now.replace(hour=KST_OPEN.hour, minute=KST_OPEN.minute,
                              second=0, microsecond=0)
    if now < today_open and now.weekday() < 5:
        return (today_open - now).total_seconds()
    # 다음 영업일 개장
    next_day = now + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day = next_day + timedelta(days=1)
    return (next_day.replace(hour=KST_OPEN.hour, minute=KST_OPEN.minute,
                              second=0, microsecond=0) - now).total_seconds()


def seconds_until_market_close(now: datetime) -> float:
    if not is_market_open(now):
        return 0.0
    close_today = now.replace(hour=KST_CLOSE.hour, minute=KST_CLOSE.minute,
                               second=0, microsecond=0)
    return max(0.0, (close_today - now).total_seconds())


# ----- main loop ----------------------------------------------------------

class GracefulExit:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self._sig)
        try:
            signal.signal(signal.SIGTERM, self._sig)
        except Exception:
            pass

    def _sig(self, *_):
        print("\n[graceful] SIGINT received — shutting down...", flush=True)
        self.shutdown = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="069500", help="ETF 종목 (default KODEX 200)")
    ap.add_argument("--duration", type=int, default=None,
                    help="지속 시간(초). 미지정 시 장 마감까지.")
    ap.add_argument("--until-close", action="store_true",
                    help="장 마감까지 (default — duration 없을 때)")
    ap.add_argument("--ignore-hours", action="store_true",
                    help="장 시간 무관 강제 실행 (테스트용)")
    ap.add_argument("--poll", type=float, default=5.0,
                    help="polling interval seconds (default 5)")
    ap.add_argument("--log", default=None,
                    help="로그 경로 (default data/kis_trading_log_<YYYYMMDD>.json)")
    ap.add_argument("--enter-bps", type=float, default=30.0,
                    help="iNAV 괴리 진입 임계 (bps, default 30)")
    ap.add_argument("--exit-bps", type=float, default=5.0,
                    help="청산 임계 (bps, default 5)")
    ap.add_argument("--max-position", type=int, default=10,
                    help="최대 포지션 (default 10주)")
    ap.add_argument("--initial-equity", type=float, default=10_000_000,
                    help="시작 자본 (default 1000만, 모의투자 기준)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("kis-paper")

    load_env()

    # ---------------- KIS setup
    cfg = KISConfig.from_env()
    if not (cfg.app_key and cfg.app_secret):
        logger.error("KIS_APP_KEY / KIS_APP_SECRET 미설정"); sys.exit(1)
    client = KISClient(cfg)
    logger.info(f"KIS env={cfg.env} dry_run={cfg.dry_run}")
    if cfg.env == "vts":
        logger.info("→ 모의투자 (가상머니). dry_run=true 면 mock 응답.")
    elif cfg.env == "prod":
        logger.warning("→ 실계좌. 매우 주의!")

    # ---------------- Strategy setup
    inav_est = InavEstimator(constituents=KODEX200_BASKET_TOP10,
                              cash_weight=CASH_WEIGHT)
    strategy = EtfInavArbitrage(
        enter_threshold=args.enter_bps / 10_000,
        exit_threshold=args.exit_bps / 10_000,
        max_position=args.max_position,
    )
    limits = RiskLimits(max_position_pct=0.20, daily_loss_pct=-0.015)
    state = RiskState(
        equity_open=args.initial_equity,
        equity_now=args.initial_equity,
        position_value=0,
    )

    # ---------------- Logging
    today = datetime.now().strftime("%Y%m%d")
    log_path = Path(args.log) if args.log else (
        ROOT / "data" / f"kis_trading_log_{today}.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_records: list[dict] = []

    # ---------------- Wait until market open if needed
    now = datetime.now()
    if not args.ignore_hours and not is_market_open(now):
        wait_s = seconds_until_market_open(now)
        logger.info(f"장 시간 아님. 다음 개장까지 {wait_s/60:.1f}분 대기...")
        if wait_s > 8 * 3600:
            logger.error("8시간 이상 대기 — 종료. --ignore-hours 로 강제 실행 가능")
            sys.exit(0)
        time.sleep(min(wait_s, 600))   # 최대 10분 대기 후 재확인

    # ---------------- Main loop
    exit_handler = GracefulExit()
    t_start = time.time()
    if args.duration:
        t_end = t_start + args.duration
    elif args.until_close or True:  # default
        # use scheduled close time
        t_end = t_start + seconds_until_market_close(datetime.now()) + 60
        if t_end == t_start + 60:    # already past close
            t_end = t_start + (args.duration or 600)
    else:
        t_end = t_start + 600

    tick = 0
    _last_constituent_prices: dict[str, float] = {}
    logger.info(f"=== KIS paper trading 시작 (until {datetime.fromtimestamp(t_end).strftime('%H:%M:%S')}) ===")

    while time.time() < t_end and not exit_handler.shutdown:
        now = datetime.now()
        try:
            # 1. ETF + basket 시세 (single threaded — quota safety)
            q = client.quote(args.symbol)
            etf_price = float(q.get("output", {}).get("stck_prpr", 0)) or 0.0
            if etf_price == 0:
                logger.warning(f"ETF 시세 0 — 스킵 (rt_cd={q.get('rt_cd')} {q.get('msg1','')})")
                tick += 1; time.sleep(args.poll); continue

            constituents = {}
            n_quote_ok = 0
            for sym in KODEX200_BASKET_TOP10:
                try:
                    sq = client.quote(sym)
                    p = float(sq.get("output", {}).get("stck_prpr", 0)) or 0.0
                    if p > 0:
                        constituents[sym] = p
                        n_quote_ok += 1
                    else:
                        # last known price fallback
                        constituents[sym] = _last_constituent_prices.get(sym, 0.0)
                except Exception as qe:
                    constituents[sym] = _last_constituent_prices.get(sym, 0.0)
                    if tick == 0:
                        logger.warning(f"  basket quote fail {sym}: {str(qe)[:80]}")
                time.sleep(0.05)  # quota safety
            # update cache for non-zero values
            for sym, p in constituents.items():
                if p > 0:
                    _last_constituent_prices[sym] = p

            # estimate iNAV only if 50%+ basket is fresh
            quote_ratio = n_quote_ok / len(KODEX200_BASKET_TOP10)
            if quote_ratio < 0.5:
                logger.warning(f"  basket coverage {quote_ratio*100:.0f}% — signal forced HOLD")
                inav = etf_price   # → dev = 0 → HOLD
                dev = 0.0
            else:
                inav = inav_est.estimate(constituents)
                dev = deviation(etf_price, inav)

            # 2. risk guards (시간 + 일일손실 + 포지션)
            ok, reason = check(state, limits, now)
            signal_taken = Signal.HOLD
            order_resp = None

            if ok and is_safe_to_trade(now):
                sig = strategy.decide(dev)
                if sig != Signal.HOLD:
                    qty = 1
                    side = sig.value.lower()
                    order_resp = client.order(
                        args.symbol, qty=qty, side=side, price=0, ord_div="01",
                    )
                    if cfg.dry_run:
                        logger.info(f"[DRY] {side} {args.symbol} qty={qty} (dev={dev*1e4:+.1f}bps)")
                    else:
                        logger.info(f"[ORDER] {side} {args.symbol} qty={qty} resp={order_resp}")
                    strategy.apply(sig)
                    signal_taken = sig
                    state.error_count = 0

            elif not ok:
                # 1분에 1회만 logging (spam 방지)
                if tick % (60 // max(1, int(args.poll))) == 0:
                    logger.info(f"risk halt: {reason}")
            elif not is_safe_to_trade(now):
                if tick % (60 // max(1, int(args.poll))) == 0:
                    logger.info(f"unsafe trading window (open/close 10min)")

            # 3. record
            rec = {
                "ts": now.isoformat(),
                "tick": tick,
                "etf_price": etf_price,
                "inav": round(inav, 2),
                "dev_bps": round(dev * 1e4, 2),
                "signal": signal_taken.value,
                "position": strategy.position,
                "order_response": order_resp,
                "risk_ok": ok,
                "risk_reason": reason,
            }
            log_records.append(rec)
            if tick % 10 == 0:
                logger.info(
                    f"t={tick} ETF={etf_price:.0f} iNAV={inav:.0f} dev={dev*1e4:+.1f}bps "
                    f"sig={signal_taken.value} pos={strategy.position}"
                )

        except Exception as e:
            state.error_count += 1
            logger.error(f"tick {tick} error: {type(e).__name__}: {str(e)[:200]}")
            if state.error_count >= limits.max_consecutive_errors * 3:
                # tolerance: 9회 (3 × 3) 까지. 일시적 quote 에러 흡수.
                logger.error("error count limit hit — graceful exit")
                break

        tick += 1

        # 장 마감 체크
        if not args.ignore_hours and not is_market_open(now):
            logger.info("장 마감 — 종료"); break

        time.sleep(args.poll)

    # ---------------- Save log
    log_path.write_text(
        json.dumps(log_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"=== 종료. {len(log_records)} ticks → {log_path}")
    if log_records:
        signals = [r["signal"] for r in log_records if r["signal"] != "HOLD"]
        logger.info(f"  체결 signal: {len(signals)} 건  (BUY {signals.count('BUY')} / SELL {signals.count('SELL')})")
        final_pos = log_records[-1]["position"]
        logger.info(f"  최종 포지션: {final_pos}주")


if __name__ == "__main__":
    main()
