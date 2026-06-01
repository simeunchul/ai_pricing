"""KIS 모의투자 실전 운영 스크립트 — 다중 종목 동시 운영.

특징:
  - 다중 KOSPI200 ETF 동시 매매 (069500/102110/148020/152100/278530)
  - 공유 바스켓 (KOSPI 200 상위 10종목) → 종목별 calibration_factor 로 iNAV 추정
  - 종목별 strategy / position 분리
  - 장 시작/종료 자동 정렬 (KST 09:00~15:30)
  - 장 시작/마감 10분 자동 차단 (risk guard)
  - 모든 tick 을 JSON 로그 (audit trail) — 종목 단위 record
  - 신호 발생 시 mock 또는 실주문 (KIS_DRY_RUN 따름)
  - 중간 SIGINT (Ctrl+C) 시 graceful shutdown + 로그 저장
  - 매 tick 즉시 atomic flush (dashboard 실시간 반영)

Usage:
  # 1. 다중 종목 (default 5개)
  python scripts/run_kis_paper_trading.py

  # 2. 사용자 종목 + 공격적 임계
  python scripts/run_kis_paper_trading.py \
      --symbols 069500,102110,148020 --enter-bps 1 --exit-bps 0.2 --max-position 20

  # 3. 시간 무시 + 일정 시간만 (장외 테스트)
  python scripts/run_kis_paper_trading.py --duration 600 --ignore-hours
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

# Windows cp949 console can't render em-dash / ≡ etc. used throughout this
# file's help strings and log messages — force UTF-8 so the runner doesn't
# die on its first localized print at 09:00.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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


# 각 ETF 의 추종 지수 상위 종목 + 근사 가중치.
# 정확 weights 는 운용사 PDF (일별) 기준이지만, InavEstimator 가 first-tick calibration 으로
# 절대 수준을 흡수하기 때문에 작은 weight 오차는 dev 에 큰 영향 없음. 변화율 추적이 핵심.
#
# 출처: 각 운용사 ETF 운용보고서 + DART 분기 공시 (2025년 말 기준 근사).

KOSPI200_BASKET_TOP10 = {
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

KOSDAQ150_BASKET_TOP10 = {
    "247540": 0.10,   # 에코프로비엠
    "086520": 0.08,   # 에코프로
    "196170": 0.07,   # 알테오젠
    "028300": 0.05,   # HLB
    "091990": 0.04,   # 셀트리온헬스케어 (주: 코스피 이전 가능성 — 시점 따라 비중 변동)
    "035900": 0.03,   # JYP Ent.
    "263750": 0.03,   # 펄어비스
    "058470": 0.03,   # 리노공업
    "214150": 0.03,   # 클래시스
    "145020": 0.03,   # 휴젤
}

SEMICONDUCTOR_BASKET_TOP10 = {
    "005930": 0.25,   # 삼성전자
    "000660": 0.20,   # SK하이닉스
    "042700": 0.05,   # 한미반도체
    "240810": 0.04,   # 원익IPS
    "357780": 0.04,   # 솔브레인
    "058470": 0.03,   # 리노공업
    "039030": 0.03,   # 이오테크닉스
    "000990": 0.03,   # DB하이텍
    "036930": 0.03,   # 주성엔지니어링
    "005290": 0.02,   # 동진쎄미켐
}

BATTERY_BASKET_TOP10 = {
    "373220": 0.20,   # LG에너지솔루션
    "247540": 0.10,   # 에코프로비엠
    "006400": 0.08,   # 삼성SDI
    "086520": 0.07,   # 에코프로
    "003670": 0.05,   # 포스코퓨처엠
    "066970": 0.04,   # 엘앤에프
    "096770": 0.04,   # SK이노베이션
    "121600": 0.03,   # 나노신소재
    "352820": 0.03,   # 하이브 (주: 일부 ETF 에 소량)
    "020150": 0.03,   # 일진머티리얼즈
}

HEALTHCARE_BASKET_TOP10 = {
    "068270": 0.18,   # 셀트리온
    "207940": 0.15,   # 삼성바이오로직스
    "128940": 0.06,   # 한미약품
    "000100": 0.05,   # 유한양행
    "326030": 0.04,   # SK바이오팜
    "302440": 0.04,   # SK바이오사이언스
    "196170": 0.04,   # 알테오젠
    "069620": 0.03,   # 대웅제약
    "185750": 0.03,   # 종근당
    "009420": 0.02,   # 한올바이오파마
}

# ETF 코드 → (이름, 바스켓 dict)
ETF_BASKETS: dict[str, tuple[str, dict[str, float]]] = {
    "069500": ("KODEX 200", KOSPI200_BASKET_TOP10),
    "102110": ("TIGER 200", KOSPI200_BASKET_TOP10),
    # "148020": ("KBSTAR 200", KOSPI200_BASKET_TOP10),  # 2026-04-28: KIS vts quote fail (저유동성)
    "152100": ("ARIRANG 200", KOSPI200_BASKET_TOP10),
    "278530": ("KODEX 200TR", KOSPI200_BASKET_TOP10),
    "105190": ("KINDEX 200", KOSPI200_BASKET_TOP10),
    "229200": ("KODEX 코스닥150", KOSDAQ150_BASKET_TOP10),
    "091160": ("KODEX 반도체", SEMICONDUCTOR_BASKET_TOP10),
    "305720": ("KODEX 2차전지산업", BATTERY_BASKET_TOP10),
    "266420": ("KODEX 헬스케어", HEALTHCARE_BASKET_TOP10),
}
SYMBOL_NAMES = {k: v[0] for k, v in ETF_BASKETS.items()}

# Inverse ETF 매핑 (개인 직접 공매도 우회용 합성 short).
# BUY 신호가 인버스 ETF에 들어가면 underlying 에 대한 short 노출이 됨.
# packages/autotrader/src/autotrader/runner.py 의 INVERSE_MAP 과 동기화 유지.
INVERSE_MAP = {
    "069500": "252670",   # KODEX 200          → KODEX 200 인버스
    "102110": "252670",   # TIGER 200          → KODEX 200 인버스 (동일 underlying)
    # "148020": "252670", # KBSTAR 200         → KODEX 200 인버스  (148020 운영 제외됨)
    "152100": "252670",   # ARIRANG 200        → KODEX 200 인버스
    "278530": "252670",   # KODEX 200TR        → KODEX 200 인버스
    "105190": "252670",   # KINDEX 200         → KODEX 200 인버스
    # "229200": "251340", # KODEX 코스닥150    → KODEX 코스닥150 선물인버스
    #                       (2026-04-28: 251340 KIS vts quote fail → 매핑 비활성)
    # 섹터 ETF (반도체, 2차전지, 헬스케어) 는 명확한 1:1 인버스가 없음 →
    # 양수 deviation 시그널 발생 시 synthetic short 스킵 (경고 로그만).
}

# Default: KOSPI 200 (069500) + 섹터 4종 → 분산 검증
DEFAULT_SYMBOLS = "069500,229200,091160,305720,266420"


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


# ----- per-symbol state ---------------------------------------------------

@dataclass
class SymbolState:
    """한 종목당 InavEstimator + Strategy 결합. 바스켓 weights 는 공유."""
    symbol: str
    name: str
    inav_est: InavEstimator
    strategy: EtfInavArbitrage


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
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS,
                    help=f"콤마 구분 ETF 종목 리스트 (default {DEFAULT_SYMBOLS})")
    ap.add_argument("--symbol", default=None,
                    help="단일 종목 (legacy, --symbols 우선)")
    ap.add_argument("--duration", type=int, default=None,
                    help="지속 시간(초). 미지정 시 장 마감까지.")
    ap.add_argument("--until-close", action="store_true",
                    help="장 마감까지 (default — duration 없을 때)")
    ap.add_argument("--ignore-hours", action="store_true",
                    help="장 시간 무관 강제 실행 (테스트용)")
    ap.add_argument("--poll", type=float, default=2.0,
                    help="symbol 루프 사이 추가 sleep 초 (default 2)")
    ap.add_argument("--quote-sleep", type=float, default=0.7,
                    help="quote 호출 사이 sleep 초 (default 0.7, vts EGW00201 회피)")
    ap.add_argument("--log", default=None,
                    help="로그 경로 (default data/kis_trading_log_<YYYYMMDD>.json)")
    ap.add_argument("--enter-bps", type=float, default=1.0,
                    help="iNAV 괴리 진입 임계 (bps, default 1)")
    ap.add_argument("--exit-bps", type=float, default=0.2,
                    help="청산 임계 (bps, default 0.2)")
    ap.add_argument("--max-position", type=int, default=20,
                    help="종목당 최대 포지션 (default 20주)")
    ap.add_argument("--qty-per-step", type=int, default=5,
                    help="deviation 한 단계(=enter-bps)당 주문 수량 (default 5, "
                         "1주씩 보수적으로 가려면 1)")
    ap.add_argument("--initial-equity", type=float, default=10_000_000,
                    help="시작 자본 (default 1000만, 모의투자 기준)")
    ap.add_argument("--cash-sync-every", type=int, default=10,
                    help="N tick 마다 KIS 잔고 재조회해서 가용현금 동기화 "
                         "(default 10, 0=매tick)")
    ap.add_argument("--cash-buffer", type=int, default=0,
                    help="가용현금에서 buffer 만큼은 매수에 안 씀. "
                         "양수=현금 비축, 음수=phantom debt 허용 한도. "
                         "default 0 → 가용현금(≈시드 천만원) 한도까지 매매, "
                         "T+2 결제 차입 금지.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("kis-paper")

    load_env()

    # ---------------- Resolve symbols
    if args.symbol and not args.symbols:
        symbols = [args.symbol]
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("종목 미지정"); sys.exit(1)
    logger.info(f"운영 종목 ({len(symbols)}): " + ", ".join(
        f"{s}({SYMBOL_NAMES.get(s, '?')})" for s in symbols
    ))

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

    # ---------------- Per-symbol state (각 ETF 자기 바스켓)
    sym_states: dict[str, SymbolState] = {}
    for sym in symbols:
        if sym not in ETF_BASKETS:
            logger.error(f"미등록 ETF {sym}: ETF_BASKETS 에 추가 필요"); sys.exit(1)
        name, basket = ETF_BASKETS[sym]
        cash_w = 1.0 - sum(basket.values())
        sym_states[sym] = SymbolState(
            symbol=sym,
            name=name,
            inav_est=InavEstimator(constituents=basket, cash_weight=cash_w),
            strategy=EtfInavArbitrage(
                enter_threshold=args.enter_bps / 10_000,
                exit_threshold=args.exit_bps / 10_000,
                qty_per_step=args.qty_per_step,
                max_position=args.max_position,
            ),
        )

    # 모든 ETF 바스켓의 union — quote 1회만 페칭
    all_constituents: set[str] = set()
    for sym in symbols:
        all_constituents.update(ETF_BASKETS[sym][1].keys())
    logger.info(
        f"바스켓 union: {len(all_constituents)} 종목 (per-tick quote)"
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

    # ---------------- Resume: load existing log if any
    log_records: list[dict] = []
    resume_tick_offset = 0
    if log_path.exists() and log_path.stat().st_size > 10:
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                log_records = existing
                resume_tick_offset = max(r.get("tick", 0) for r in existing) + 1
                logger.info(
                    f"resume: loaded {len(existing)} prior ticks "
                    f"(last tick={resume_tick_offset - 1}); appending from tick={resume_tick_offset}"
                )
        except Exception as e:
            logger.warning(f"resume failed (corrupt log?): {e}; starting fresh")
            log_records = []

    # ---------------- Cash guard helpers
    def _query_implied_cash() -> tuple[float, dict[str, float]]:
        """KIS 잔고에서 (가용현금, 보유 평가단가dict) 반환.
        가용현금 = 총평가 - 보유평가합 (음수면 phantom debt).
        실패 시 (None, {}) 반환 — 호출자가 이전 값 유지하도록.
        """
        try:
            b = client.balance()
            if b.get("rt_cd") != "0":
                return None, {}
            o2 = b.get("output2", [{}])[0] if b.get("output2") else {}
            o1 = b.get("output1") or []
            tot = int(o2.get("tot_evlu_amt", 0))
            sum_h = sum(int(h.get("evlu_amt", 0)) for h in o1
                        if int(h.get("hldg_qty", 0)) > 0)
            cur_prices = {h.get("pdno"): float(h.get("prpr", 0))
                          for h in o1 if int(h.get("hldg_qty", 0)) > 0}
            return float(tot - sum_h), cur_prices
        except Exception as e:
            logger.debug(f"cash query fail: {e}")
            return None, {}

    available_cash = 0.0  # 음수 가능
    last_cash_sync_tick = -10**9

    # ---------------- Sync per-symbol position from KIS holdings
    try:
        bal = client.balance()
        if bal.get("rt_cd") == "0":
            holdings = bal.get("output1", []) or []
            held = {h.get("pdno"): int(h.get("hldg_qty", 0)) for h in holdings}
            for sym, ss in sym_states.items():
                qty_held = held.get(sym, 0)
                if qty_held != ss.strategy.position:
                    logger.info(
                        f"position sync [{sym}]: KIS={qty_held}, "
                        f"strategy={ss.strategy.position} → updating"
                    )
                    ss.strategy.position = qty_held
                # 인버스 ETF (합성 short leg) 도 같이 동기화
                inv_sym = INVERSE_MAP.get(sym)
                if inv_sym:
                    inv_held = held.get(inv_sym, 0)
                    if inv_held != ss.strategy.inverse_position:
                        logger.info(
                            f"inverse sync [{sym}/{inv_sym}]: KIS={inv_held}, "
                            f"strategy={ss.strategy.inverse_position} → updating"
                        )
                        ss.strategy.inverse_position = inv_held
            out2 = bal.get("output2", [{}])[0] if bal.get("output2") else {}
            tot_eval = int(out2.get("tot_evlu_amt", 0))
            if tot_eval > 0:
                state.equity_now = float(tot_eval)
                logger.info(f"equity sync: {tot_eval:,} KRW")
    except Exception as e:
        logger.warning(f"balance sync failed: {e}; using defaults")

    # 초기 가용현금 측정 (위 sync 시 한 번 더 호출하지 않도록 직접 계산)
    try:
        if 'bal' in locals() and bal.get("rt_cd") == "0":
            o2 = bal.get("output2", [{}])[0] if bal.get("output2") else {}
            o1 = bal.get("output1") or []
            tot0 = int(o2.get("tot_evlu_amt", 0))
            sum_h0 = sum(int(h.get("evlu_amt", 0)) for h in o1
                         if int(h.get("hldg_qty", 0)) > 0)
            available_cash = float(tot0 - sum_h0)
            last_cash_sync_tick = 0
            logger.info(
                f"가용현금 초기값: {available_cash:+,.0f}원 "
                f"(총평가 {tot0:,} - 보유평가 {sum_h0:,})"
            )
            if available_cash < 0:
                logger.warning(
                    f"⚠ phantom margin debt 감지 ({available_cash:+,.0f}원) — "
                    f"수렴 시 자연 청산 또는 emergency_liquidate 권장"
                )
    except Exception as e:
        logger.warning(f"initial cash measure failed: {e}")

    def _flush_log() -> None:
        """Atomic write to avoid mid-read corruption."""
        tmp = log_path.with_suffix(log_path.suffix + ".tmp")
        tmp.write_text(json.dumps(log_records, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        tmp.replace(log_path)

    # ---------------- Wait until market open if needed
    now = datetime.now()
    if not args.ignore_hours and not is_market_open(now):
        wait_s = seconds_until_market_open(now)
        logger.info(f"장 시간 아님. 다음 개장까지 {wait_s/60:.1f}분 대기...")
        if wait_s > 8 * 3600:
            logger.error("8시간 이상 대기 — 종료. --ignore-hours 로 강제 실행 가능")
            sys.exit(0)
        time.sleep(min(wait_s, 600))

    # ---------------- Main loop
    exit_handler = GracefulExit()
    t_start = time.time()
    if args.duration:
        t_end = t_start + args.duration
    elif args.until_close or True:
        t_end = t_start + seconds_until_market_close(datetime.now()) + 60
        if t_end == t_start + 60:
            t_end = t_start + (args.duration or 600)
    else:
        t_end = t_start + 600

    tick = resume_tick_offset
    _last_constituent_prices: dict[str, float] = {}
    _last_etf_prices: dict[str, float] = {}
    _last_inverse_prices: dict[str, float] = {}
    unique_inverse_codes = set(INVERSE_MAP.values())
    logger.info(
        f"=== KIS paper trading 시작 (until "
        f"{datetime.fromtimestamp(t_end).strftime('%H:%M:%S')}, "
        f"enter={args.enter_bps}bps exit={args.exit_bps}bps maxpos={args.max_position}) ==="
    )

    while time.time() < t_end and not exit_handler.shutdown:
        now = datetime.now()
        try:
            # 1. ETF 시세 (전 종목 일괄)
            etf_prices: dict[str, float] = {}
            for sym in symbols:
                try:
                    q = client.quote(sym)
                    p = float(q.get("output", {}).get("stck_prpr", 0)) or 0.0
                    if p > 0:
                        etf_prices[sym] = p
                        _last_etf_prices[sym] = p
                    else:
                        etf_prices[sym] = _last_etf_prices.get(sym, 0.0)
                except Exception as qe:
                    etf_prices[sym] = _last_etf_prices.get(sym, 0.0)
                    if tick == 0:
                        logger.warning(f"  ETF quote fail {sym}: {str(qe)[:80]}")
                time.sleep(args.quote_sleep)

            # 1.5 인버스 ETF 시세 — cash guard 시 정확한 cost 계산용
            inverse_prices: dict[str, float] = {}
            for inv_sym in unique_inverse_codes:
                try:
                    iq = client.quote(inv_sym)
                    ip = float(iq.get("output", {}).get("stck_prpr", 0)) or 0.0
                    if ip > 0:
                        inverse_prices[inv_sym] = ip
                        _last_inverse_prices[inv_sym] = ip
                    else:
                        inverse_prices[inv_sym] = _last_inverse_prices.get(inv_sym, 0.0)
                except Exception as qe:
                    inverse_prices[inv_sym] = _last_inverse_prices.get(inv_sym, 0.0)
                    if tick == resume_tick_offset:
                        logger.warning(f"  inverse quote fail {inv_sym}: {str(qe)[:80]}")
                time.sleep(args.quote_sleep)

            # 1.6 가용현금 주기적 동기화 (KIS 잔고 재조회)
            if (args.cash_sync_every == 0
                    or tick - last_cash_sync_tick >= args.cash_sync_every):
                cash_new, _ = _query_implied_cash()
                if cash_new is not None:
                    if abs(cash_new - available_cash) > 1000:
                        logger.info(
                            f"cash sync: {available_cash:+,.0f} → {cash_new:+,.0f} "
                            f"(drift {cash_new - available_cash:+,.0f})"
                        )
                    available_cash = cash_new
                    last_cash_sync_tick = tick

            # 2. 바스켓 시세 — union 한 번만 페칭 (모든 ETF 가 공유)
            constituents: dict[str, float] = {}
            n_quote_ok = 0
            for csym in all_constituents:
                try:
                    sq = client.quote(csym)
                    p = float(sq.get("output", {}).get("stck_prpr", 0)) or 0.0
                    if p > 0:
                        constituents[csym] = p
                        n_quote_ok += 1
                    else:
                        constituents[csym] = _last_constituent_prices.get(csym, 0.0)
                except Exception as qe:
                    constituents[csym] = _last_constituent_prices.get(csym, 0.0)
                    if tick == 0:
                        logger.warning(f"  basket quote fail {csym}: {str(qe)[:80]}")
                time.sleep(args.quote_sleep)
            for csym, p in constituents.items():
                if p > 0:
                    _last_constituent_prices[csym] = p

            quote_ratio = n_quote_ok / max(1, len(all_constituents))

            # 3. risk guard (시간 + 일일손실)
            ok, reason = check(state, limits, now)

            # 4. 종목별 평가/매매 — 각 ETF 가 자기 바스켓의 부분집합으로 iNAV 계산
            for sym in symbols:
                ss = sym_states[sym]
                etf_price = etf_prices.get(sym, 0.0)
                if etf_price <= 0:
                    continue

                # ETF 자기 바스켓 sub-prices + 100% 커버리지 강제
                # (부분 커버리지로 calibration 하면 factor 가 잘못 잡혀
                # 추후 누락 leg 복귀 시 iNAV 가 인위적으로 점프 → 잘못된 신호.
                # 이 버그를 막기 위해 calibration/estimation 둘 다 모든 leg 필수.)
                own_basket = ETF_BASKETS[sym][1]
                sub_prices = {k: constituents.get(k, 0.0) for k in own_basket}
                own_n_nonzero = sum(1 for p in sub_prices.values() if p > 0)
                own_full_coverage = own_n_nonzero == len(own_basket)

                if not ss.inav_est.calibrated:
                    if not own_full_coverage:
                        if tick == resume_tick_offset or tick % 30 == 0:
                            logger.info(
                                f"  [{sym}] calibration deferred: "
                                f"{own_n_nonzero}/{len(own_basket)} legs (need 100%)"
                            )
                        inav = etf_price; dev = 0.0
                    else:
                        factor = ss.inav_est.calibrate(etf_price, sub_prices)
                        if factor > 0:
                            logger.info(
                                f"iNAV calibrated [{sym} {ss.name}]: factor={factor:.4f} "
                                f"(etf={etf_price:.0f}, n_legs={own_n_nonzero}/{len(own_basket)} ✓)"
                            )
                        inav = etf_price; dev = 0.0
                else:
                    # estimate() 가 strict 모드 — frozen leg 누락 시 0 반환
                    inav = ss.inav_est.estimate(sub_prices)
                    if inav <= 0:
                        if tick % 30 == 0:
                            logger.info(
                                f"  [{sym}] iNAV unavailable — "
                                f"calibrated leg missing this tick (skip)"
                            )
                        inav = etf_price; dev = 0.0
                    else:
                        dev = deviation(etf_price, inav)

                signal_taken = Signal.HOLD
                order_resp = None
                order_qty = 0
                order_target = None
                order_inverse = False
                cash_skip = False
                if ok and is_safe_to_trade(now):
                    sig, qty, use_inverse = ss.strategy.decide_aggressive(dev)
                    if sig != Signal.HOLD and qty > 0:
                        target_sym = INVERSE_MAP.get(sym) if use_inverse else sym
                        target_price = (inverse_prices.get(target_sym, 0.0)
                                        if use_inverse else etf_price)
                        skip_reason = None

                        if use_inverse and target_sym is None:
                            skip_reason = "no inverse mapping"
                        elif sig == Signal.BUY:
                            # CASH GUARD — phantom margin debt 차단
                            if target_price <= 0:
                                skip_reason = f"no price for {target_sym}"
                            else:
                                spendable = available_cash - args.cash_buffer
                                est_cost = qty * target_price
                                if est_cost > spendable:
                                    max_qty = (int(spendable // target_price)
                                               if target_price > 0 else 0)
                                    if max_qty <= 0:
                                        cash_skip = True
                                        skip_reason = (
                                            f"cash {available_cash:+,.0f}원 부족 "
                                            f"(필요 {est_cost:,.0f}원, "
                                            f"buffer 후 {spendable:+,.0f}원)"
                                        )
                                    else:
                                        logger.info(
                                            f"  [{sym}] cash 제한: qty {qty}→{max_qty} "
                                            f"(가용 {spendable:+,.0f}원)"
                                        )
                                        qty = max_qty

                        if skip_reason:
                            if cash_skip:
                                # spam 방지: 5tick 마다 한 번만 로그
                                if tick % 5 == 0:
                                    logger.warning(f"  [{sym}] BUY skip: {skip_reason}")
                            else:
                                logger.warning(f"  [{sym}] skip: {skip_reason}")
                        else:
                            side = sig.value.lower()
                            order_resp = client.order(
                                target_sym, qty=qty, side=side,
                                price=0, ord_div="01",
                            )
                            leg = "INV" if use_inverse else "ETF"
                            if cfg.dry_run:
                                logger.info(
                                    f"[DRY] {side} [{target_sym}] ({leg}) "
                                    f"qty={qty} dev={dev*1e4:+.2f}bps"
                                )
                            else:
                                logger.info(
                                    f"[ORDER] {side} [{target_sym}] ({leg}) "
                                    f"qty={qty} dev={dev*1e4:+.2f}bps "
                                    f"cash={available_cash:+,.0f} "
                                    f"resp={order_resp}"
                                )
                            ss.strategy.apply_aggressive(sig, qty, use_inverse)
                            signal_taken = sig
                            order_qty = qty
                            order_target = target_sym
                            order_inverse = use_inverse
                            # 로컬 가용현금 추정 갱신 (다음 sync 까지 임시값)
                            est_cost = qty * target_price
                            if sig == Signal.BUY:
                                available_cash -= est_cost
                            else:  # SELL
                                available_cash += est_cost
                            state.error_count = 0

                rec = {
                    "ts": now.isoformat(),
                    "tick": tick,
                    "symbol": sym,
                    "name": ss.name,
                    "etf_price": etf_price,
                    "inav": round(inav, 2),
                    "dev_bps": round(dev * 1e4, 2),
                    "signal": signal_taken.value,
                    "order_qty": order_qty,
                    "order_target": order_target,
                    "order_inverse": order_inverse,
                    "etf_position": ss.strategy.position,
                    "inverse_position": ss.strategy.inverse_position,
                    "position": ss.strategy.position,  # 하위 호환 (대시보드/이전 분석)
                    "order_response": order_resp,
                    "risk_ok": ok,
                    "risk_reason": reason,
                }
                log_records.append(rec)

            # 5. spam-suppressed risk halt log
            if not ok and tick % max(1, int(60 // max(1, args.poll))) == 0:
                logger.info(f"risk halt: {reason}")

            _flush_log()
            if tick % 5 == 0:
                summary = " | ".join(
                    f"{s}={etf_prices.get(s, 0):.0f}@p{sym_states[s].strategy.position}"
                    for s in symbols
                )
                logger.info(f"t={tick} basket_cov={quote_ratio*100:.0f}% {summary}")

        except Exception as e:
            state.error_count += 1
            logger.error(f"tick {tick} error: {type(e).__name__}: {str(e)[:200]}")
            if state.error_count >= limits.max_consecutive_errors * 3:
                logger.error("error count limit hit — graceful exit")
                break

        tick += 1

        if not args.ignore_hours and not is_market_open(now):
            logger.info("장 마감 — 종료"); break

        time.sleep(args.poll)

    # ---------------- Save log
    log_path.write_text(
        json.dumps(log_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"=== 종료. {len(log_records)} records → {log_path}")
    if log_records:
        for sym in symbols:
            sub = [r for r in log_records if r.get("symbol") == sym]
            sigs = [r["signal"] for r in sub if r["signal"] != "HOLD"]
            n_buy = sum(1 for s in sigs if s == "BUY")
            n_sell = sum(1 for s in sigs if s == "SELL")
            tot_qty = sum(r.get("order_qty", 0) for r in sub)
            n_inv = sum(1 for r in sub if r.get("order_inverse"))
            last = sub[-1] if sub else {}
            final_etf = last.get("etf_position", last.get("position", 0))
            final_inv = last.get("inverse_position", 0)
            logger.info(
                f"  [{sym}] {SYMBOL_NAMES.get(sym, '?')}: "
                f"records={len(sub)} BUY={n_buy} SELL={n_sell} "
                f"체결수량합={tot_qty}주 (그중 INV={n_inv}건) "
                f"최종 ETF={final_etf}주 / INV={final_inv}주"
            )


if __name__ == "__main__":
    main()
