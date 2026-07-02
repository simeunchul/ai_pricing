"""Dual Confirmation 동적 universe 모의투자 운용 스크립트.

매일 1회 실행 (장 시작 직후 권장):
  1. KIS 토큰 + 잔고 확인
  2. 외국인+기관 가집계 호출 (매수/매도 상위 60종)
  3. Dual confirmation 통과 종목 식별
  4. MDD cap 체크 (전일 종가 기준)
  5. 보유 중 매도 후보 → 매도 주문
  6. 새 매수 후보 → 매수 주문 (max_concurrent 한도)
  7. 상태 저장 + 로그 기록

운용 룰 (백테스트 검증된 best):
  - enter_threshold = 0.05 (외인+기관 둘 다 ±5%)
  - max_concurrent = 7 종목
  - mdd_cap = 30% / cooldown 10일
  - 균등 자본 분산

Phase 1 (오늘): --dry-only 플래그로 신호만 출력, 실제 주문 X
Phase 2 (검증 후): --dry-only 제거 → 실제 KIS 주문
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))

from autotrader.broker.kis_client import KISClient, KISConfig
from autotrader.market.foreign_inst_realtime import (
    parse_foreign_institution_response, compute_flow_ratios,
    PersistentPayloadCache,
)
from autotrader.market.liquidity_filter import (
    evaluate_liquidity, MIN_TRADING_VALUE_KRW, MAX_SPREAD_BPS,
)
from autotrader.paper.dual_state import DualPaperState, Position
from autotrader.data.krx_investor import ensure_combined


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# 운용 파라미터 (백테스트 best from sweep_dynamic_dual)
ENTER_THRESHOLD = 0.05
MAX_CONCURRENT = 7
# sweep_risk_controls (2026-04-28): 25% 가 best (return +18.77%, MDD -28.5%,
# Calmar 0.129) — 30% 보다 모든 지표 우수해서 25% 로 전환.
MDD_CAP = 0.25
# cooldown 은 백테스트에서 sweep 된 적이 없어 검증 X — 운영도 cooldown 없이
# (백테스트 결과와 동일 조건 유지). MDD CAP 발동 후 다음 iter 부터 매수 재개.
COOLDOWN_DAYS = 0
MIN_SIZING_DIVISOR = 3   # cash-aware sizing 최소 분산 (한 종목 올인 방지)
MIN_TRADING_VALUE = MIN_TRADING_VALUE_KRW   # 5억원 (Phase 1)
MAX_SPREAD = MAX_SPREAD_BPS                  # 30bps = 0.3% (Phase 2)

# ===== WF-검증 추가 룰 (2026-06-01 적용) =====
# E2 비대칭 매도: 매수 +5% 보다 매도 -7% 가 더 강한 신호 요구 → over-trade 방지.
# 백테스트(sell rule sweep) 에서 Calmar 0.49 → 0.59, 거래수 499 → 358 검증.
SELL_THRESHOLD = 0.07

# 60일 모멘텀 confluence: 외인+기관 매수 + 60일 가격 추세 양수일 때만 진입.
# falling-knife 함정(추세 하락인데 flow만 양) 차단. 백테스트 Calmar 0.59 → 0.72,
# OOS 검증에서도 모든 파라미터 조합이 baseline 위.
MOMENTUM_LOOKBACK = 60
MOMENTUM_MIN = 0.0

# Replacement 룰: 슬롯 가득 + 새 매수 후보 있을 때 "보유 N일+, 손익 X% 미만"인
# 횡보·미미한 손익 포지션을 다음날 시가에 청산해 자리 확보.
# WF 검증: hold≥7일 압도적 수렴, pnl_max는 1~2% 영역이 robust. 보수적으로 2% 채택.
REPLACEMENT_PNL_MAX = 0.02
REPLACEMENT_MIN_HOLD = 7

# A3 매도 룰: 단기 수익률 음전 + 손실 영역 (30위 밖 가시성 사각지대 backup).
# 보유 종목의 N일 가격 수익률 < threshold AND 현재가 < 진입가 → 매도.
# 백테스트(2026-06-05): Calmar 0.85→1.02, MDD -45%→-38%, bear +20%→+28%.
# 추세분석(EMA)과 TP는 backtest에서 더 나빴음 — A3가 데이터로 검증된 winner.
SHORT_MOM_LOOKBACK = 10
SHORT_MOM_THRESHOLD = -0.05

# 매수 단기추세 가드 (Fix 1, 2026-06-09): 매수 후보의 SHORT_MOM_LOOKBACK(10)일
# 수익률이 이 값 미만이면 매수 보류. A3 매도가 즉시 뱉어낼 falling-knife(60일은 +
# 인데 10일은 급락 중인 종목)를 애초에 안 사 churn 차단. A3 매도 룰 자체는 불변.
# 백테스트(4년 40종): churn -18%, 수익 +205%→+225%, Calmar 0.63→0.69.
BUY_SHORT_MOM_THRESHOLD = -0.05


def _append_trade(state_path: Path, *, ts: str, side: str, code: str,
                   name: str, qty: int, price: float,
                   avg_entry: float | None = None,
                   pnl_pct: float | None = None,
                   pnl_krw: float | None = None,
                   flow: float | None = None,
                   inst: float | None = None,
                   reason: str = "") -> None:
    """매매 1건을 dual_trades.parquet 에 row-level append.

    - log 텍스트 파싱과 무관하게 정확한 매도 PnL 계산 가능 (avg_entry 보존).
    - dashboard 는 이 파일을 우선 source 로 사용 (없으면 log 파싱 fallback).
    """
    import pandas as pd
    p = state_path.parent / "dual_trades.parquet"
    row = {
        "timestamp": ts, "side": side, "code": code, "name": name,
        "qty": int(qty), "price": float(price),
        "avg_entry": float(avg_entry) if avg_entry is not None else None,
        "pnl_pct": float(pnl_pct) if pnl_pct is not None else None,
        "pnl_krw": float(pnl_krw) if pnl_krw is not None else None,
        "flow": float(flow) if flow is not None else None,
        "inst": float(inst) if inst is not None else None,
        "reason": reason,
    }
    try:
        if p.exists():
            df = pd.read_parquet(p)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_parquet(p)
    except Exception:
        pass   # 기록 실패해도 매매는 진행 (log 가 fallback)


# Circuit breaker — KIS API 일시 장애 대응
ERROR_WINDOW_SEC = 600      # 10분 윈도우
ERROR_HALT_THRESHOLD = 10   # 10분 내 API 에러 10회 이상 → 매매 일시 차단
HALT_COOLDOWN_SEC = 300     # 차단 후 5분 정지 → 자동 해제 (다음 iter 재평가)
_api_error_history: list[float] = []
_halted_until: float = 0.0


def _persist_circuit_status(state_path_parent: Path) -> None:
    """API 에러 / circuit halt 상태를 dashboard 가 읽도록 JSON 으로 저장."""
    import json
    try:
        p = state_path_parent / "dual_circuit.json"
        import time as _t
        now = _t.time()
        p.write_text(json.dumps({
            "errors_in_window": len(_api_error_history),
            "window_sec": ERROR_WINDOW_SEC,
            "threshold": ERROR_HALT_THRESHOLD,
            "halted": _halted_until > now,
            "halted_remaining_sec": max(0, int(_halted_until - now)),
            "last_error_ts": _api_error_history[-1] if _api_error_history else None,
            "updated": now,
        }, indent=2), encoding="utf-8")
    except Exception:
        pass


def _record_api_error(log) -> bool:
    """API 에러 1건 기록. 임계 도달하면 매매 차단 (True 반환)."""
    import time as _t
    now = _t.time()
    _api_error_history.append(now)
    # window 밖 에러 제거
    while _api_error_history and _api_error_history[0] < now - ERROR_WINDOW_SEC:
        _api_error_history.pop(0)
    triggered = False
    if len(_api_error_history) >= ERROR_HALT_THRESHOLD:
        global _halted_until
        _halted_until = now + HALT_COOLDOWN_SEC
        log.error(
            f"[circuit] API 에러 {len(_api_error_history)}회 / "
            f"{ERROR_WINDOW_SEC}s — 매매 {HALT_COOLDOWN_SEC}s 차단"
        )
        _api_error_history.clear()
        triggered = True
    # state dir 로 dump (dashboard 에서 읽음)
    try:
        from pathlib import Path as _P
        _persist_circuit_status(_P("data"))
    except Exception:
        pass
    return triggered


def _is_halted(log) -> bool:
    import time as _t
    if _halted_until > _t.time():
        remaining = int(_halted_until - _t.time())
        log.warning(f"[circuit] 매매 차단 중 — 해제까지 {remaining}s")
        return True
    return False


def safe_order(client, sym: str, qty: int, side: str, log,
               max_retry: int = 1):
    """KIS 주문 wrapper — rt_cd 검증 + 500/timeout 시 idempotency 확인 후 1회 retry.

    Returns:
        (status, resp) where status ∈ {'ok', 'rejected', 'failed'}
          ok       — 주문 정상 (rt_cd 0 또는 잔고 변화 확인)
          rejected — KIS 가 명시적으로 거부 (rt_cd != 0). retry 무의미.
          failed   — 통신 실패 + 잔고 변화 없음. state 갱신 X.
    """
    import time as _t

    def _check_landed() -> bool:
        """500 후 잔고 확인 — 의도 주문이 들어갔으면 True."""
        try:
            bal = client.balance()
            if bal.get("rt_cd") != "0":
                return False
            out1 = bal.get("output1") or []
            for h in out1:
                if h.get("pdno") != sym:
                    continue
                kis_qty = int(str(h.get("hldg_qty") or "0").replace(",", ""))
                if side == "buy" and kis_qty >= qty:
                    return True
                if side == "sell" and kis_qty == 0:
                    return True
            return False
        except Exception:
            return False

    for attempt in range(max_retry + 1):
        try:
            resp = client.order(sym, qty=qty, side=side, price=0, ord_div="01")
            if isinstance(resp, dict) and resp.get("rt_cd") == "0":
                return ("ok", resp)
            # 명시적 reject — 잘못된 요청. retry 무의미.
            log.error(f"    [{sym}] order rejected: rt_cd="
                      f"{resp.get('rt_cd') if isinstance(resp, dict) else '?'} "
                      f"{resp.get('msg1','') if isinstance(resp, dict) else resp}")
            return ("rejected", resp)
        except Exception as e:
            log.warning(f"    [{sym}] order attempt {attempt+1} 통신 실패: {str(e)[:120]}")
            _record_api_error(log)
            # KIS vts 처리 지연 흡수 — 2초로는 부족했던 사례(2026-06-05) 관찰.
            # 첫 check 2s 후 시도, 안 보이면 8초 더 기다리고 한 번 더 (총 ~10s).
            # idempotent OK 판정되면 즉시 종료. 너무 빨리 "잔고 변화 없음"으로
            # 판정해 retry 보낸 결과 같은 주문이 두 번 들어가는 위험 차단.
            _t.sleep(2.0)
            if _check_landed():
                log.info(f"    [{sym}] 잔고 확인 — 500 났지만 주문은 들어감 (idempotent OK, 2s)")
                return ("ok", None)
            _t.sleep(8.0)
            if _check_landed():
                log.info(f"    [{sym}] 잔고 확인 — 10s 지연 후 확인, 주문 처리됨 (idempotent OK)")
                return ("ok", None)
            if attempt < max_retry:
                log.info(f"    [{sym}] 잔고 변화 없음 — retry {attempt+2}/{max_retry+1}")
                _t.sleep(1.0)
            else:
                log.error(f"    [{sym}] 최종 실패 — state 갱신 안 함")
    return ("failed", None)


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


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # TimedRotatingFileHandler — 매일 자정 회전 + 7일 보관
    # dual_trading.log → dual_trading.log.2026-05-18 식으로 archived
    from logging.handlers import TimedRotatingFileHandler
    handler = TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=7,
        encoding="utf-8", utc=False,
    )
    handler.suffix = "%Y-%m-%d"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            handler,
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_symbol_config() -> dict:
    """config/symbols.json 로드. 없으면 기본값 (모든 top 60 허용)."""
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "symbols.json"
    if not cfg_path.exists():
        return {"stock": {"whitelist": [], "blacklist": [], "max_concurrent": 7, "market_code": "0001"}}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {"stock": {"whitelist": [], "blacklist": [], "max_concurrent": 7, "market_code": "0001"}}


def apply_symbol_filter(df, whitelist: list, blacklist: list):
    """whitelist (있으면 only) + blacklist (제외) 필터. 빈 df 안전."""
    if df is None or df.empty or "symbol" not in df.columns:
        return df.copy() if df is not None else df
    if whitelist:
        df = df[df["symbol"].isin(whitelist)]
    if blacklist:
        df = df[~df["symbol"].isin(blacklist)]
    return df.copy()


def _safe_fetch_with_retry(fetch_fn, label: str, max_retries: int = 3,
                            base_backoff: float = 1.5):
    """KIS API 호출을 backoff retry. 최종 실패 시 None 반환 (caller 가 빈 처리)."""
    import logging
    log = logging.getLogger(__name__)
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn()
        except Exception as e:
            msg = str(e)[:120]
            if attempt < max_retries:
                wait = base_backoff * (2 ** (attempt - 1))   # 1.5, 3, 6 s
                log.warning(f"  [retry {attempt}/{max_retries}] {label} 실패 — {wait:.1f}s 후 재시도: {msg}")
                time.sleep(wait)
            else:
                log.error(f"  [retry FINAL] {label} 최종 실패 ({max_retries}회): {msg}")
    return None


FALLBACK_THRESHOLD_MULT = 1.5   # stale 데이터는 1.5배 강한 신호에만 진입

# 강한 신호 다발 시 포지션 cap 일시 확장 (#3)
# 외국인+기관 동시매수 강도 (min(flow, inst)) 가 이 임계 이상인 후보를 "strong"
# 으로 분류. cap 이 도달했어도 strong 후보가 free_slots 보다 많으면 cap 을
# 일시 확장해 알파 캡처. 다음 iter 에 자동 복귀 (state 에 저장 안 함).
STRONG_SIGNAL_MULT = 1.5
STRONG_THRESHOLD = ENTER_THRESHOLD * STRONG_SIGNAL_MULT   # 0.075 = 7.5%
MAX_CAP_EXTENSION = 3   # cap 7 → 최대 10 까지 확장 허용 (폭주 방지)

# 코스피 대비 underperform 강제 청산 (#5)
# 보유 종목이 매수 시점 이후 코스피 대비 이 임계 이상으로 underperform 하면
# 다음 iter 매도 단계에서 강제 청산. 약한 베타 + 그날 알파 부족 종목 자동 제거.
UNDERPERFORM_THRESHOLD = 0.05         # 코스피 대비 -5% 초과 underperform
UNDERPERFORM_MIN_HOLD_DAYS = 2        # 최소 2거래일 보유 후 검사 (잡음 흡수)
KOSPI_INDEX_CODE = "0001"             # KOSPI 종합지수 (KIS index_price)


def _parse_kospi_quote(payload: dict) -> float:
    """KIS index_price 응답 → 현재 지수값. 실패 시 0.0."""
    output = payload.get("output") if isinstance(payload, dict) else None
    if not output:
        return 0.0
    raw = output.get("bstp_nmix_prpr") if isinstance(output, dict) else None
    if raw is None:
        return 0.0
    try:
        return float(str(raw).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _days_between(entry_date_iso: str, today: datetime | None = None) -> int:
    """entry_date 와 오늘 사이 일수 추정 (단순 calendar day - 비거래일 무시).

    "unknown" 의미: KIS balance 의 thdt_buyqty=0 AND bfdy_buy_qty=0 → 그저께 이전
    매수. 정확한 날짜는 모르지만 **최소 2영업일 이상 보유 확정**. 이전 버그
    (2026-06-05 발견): "unknown" 을 0일로 반환해서 replacement / underperform
    청산 룰의 min_hold_days 게이트를 통과 못하게 만들어, 6주 보유한 종목이
    "어제 산 신규" 로 처리됨. 사용자가 호소한 "흐지부지한 주식 못 정리" 의 원인.
    수정: "unknown" 은 보수적으로 큰 값(999) 반환 → 보유일 게이트는 통과,
    상대 손익 게이트는 그대로 작동 → 적격성 판정 책임을 손익에 위임.
    """
    if not entry_date_iso:
        return 0
    if entry_date_iso == "unknown":
        return 999  # 충분히 오래 보유한 것으로 간주
    try:
        entry = datetime.fromisoformat(entry_date_iso).date()
    except ValueError:
        return 0
    today_d = (today or datetime.now()).date()
    return max(0, (today_d - entry).days)


# on-demand 모멘텀 보강(②) 파라미터. ①(refresh_momentum_cache.py)이 평소 캐시를
# 채우는 주력이고, 캐시 미스/stale 종목만 이 함수가 그 자리에서 1회 네이버에서 받아
# 캐시에 적재한다. 실패 종목은 쿨다운 동안 재시도하지 않아 폭락장 지연/네이버 부담을 막는다.
_MOM_STALE_DAYS = 3                 # 캐시 마지막 거래일이 이보다 오래되면 stale → 재수집
_MOM_ONDEMAND_MAX_PAGES = 12       # on-demand 재수집 깊이(≈120영업일, 60일 룩백에 충분)
_MOM_FETCH_COOLDOWN_SEC = 1800     # 재수집 실패 종목 재시도 쿨다운(30분)
_MOM_FAILED_AT: dict[str, float] = {}


def _get_momentum(sym: str, lookback: int = MOMENTUM_LOOKBACK) -> float | None:
    """N일(default 60) 가격 수익률 반환 — data/krx_cache/_combined.parquet 사용.

    캐시가 신선하면 그대로, 없거나 stale 하면 on-demand 로 1회 재수집(②)한다.
    재수집도 실패하면 None → caller 는 보수적으로 매수 차단(모멘텀 미확인 종목은
    진입 안 함). 평소 캐시 적재는 refresh_momentum_cache.py(①)가 담당.
    """
    now = time.time()
    failed_at = _MOM_FAILED_AT.get(sym)
    in_cooldown = failed_at is not None and (now - failed_at) < _MOM_FETCH_COOLDOWN_SEC
    try:
        df, action = ensure_combined(
            sym,
            max_age_days=_MOM_STALE_DAYS,
            min_rows=lookback + 1,
            max_pages=_MOM_ONDEMAND_MAX_PAGES,
            allow_fetch=not in_cooldown,
        )
    except Exception as e:
        _MOM_FAILED_AT[sym] = now
        logging.getLogger(__name__).warning(
            f"  [모멘텀] {sym} 재수집 예외: {type(e).__name__} — 차단(쿨다운)"
        )
        return None

    if action == "failed":
        _MOM_FAILED_AT[sym] = now
        logging.getLogger(__name__).warning(
            f"  [모멘텀] {sym} 데이터 없음 — 재수집 실패(쿨다운 {_MOM_FETCH_COOLDOWN_SEC//60}분)"
        )
        return None
    if action in ("stale", "missing"):
        return None  # 쿨다운 중 — 조용히 차단
    if action == "refetched":
        _MOM_FAILED_AT.pop(sym, None)
        logging.getLogger(__name__).info(f"  [모멘텀] {sym} on-demand 캐시 보강 완료")
    else:  # fresh
        _MOM_FAILED_AT.pop(sym, None)

    try:
        if df is None or df.empty or len(df) < lookback + 1:
            return None
        close = df["close"].iloc[-(lookback + 1):].values
        if close[0] <= 0:
            return None
        return float((close[-1] - close[0]) / close[0])
    except Exception:
        return None


def should_force_sell_underperform(
    position,
    current_price: float,
    current_kospi: float,
    today: datetime | None = None,
    threshold: float = UNDERPERFORM_THRESHOLD,
    min_hold_days: int = UNDERPERFORM_MIN_HOLD_DAYS,
) -> tuple[bool, float | None]:
    """매수 시점 대비 코스피 underperform 판정.

    Returns: (should_sell, relative_return) — relative_return 는 (종목수익률 - KOSPI수익률).
              데이터 부족 시 (False, None).

    조건:
      - position.kospi_at_entry 필요 (None 이면 skip — 기존 보유 종목 backward compat)
      - 최소 min_hold_days 거래일 경과 (잡음 흡수)
      - relative_return < -threshold 면 청산
    """
    if position.kospi_at_entry is None or position.kospi_at_entry <= 0:
        return False, None
    if current_kospi <= 0 or current_price <= 0 or position.avg_entry <= 0:
        return False, None
    if _days_between(position.entry_date, today) < min_hold_days:
        return False, None

    stock_return = (current_price - position.avg_entry) / position.avg_entry
    kospi_return = (current_kospi - position.kospi_at_entry) / position.kospi_at_entry
    relative = stock_return - kospi_return
    return (relative < -threshold), relative


def compute_effective_cap(buy_cands, positions: dict, cfg_max_concurrent: int,
                            fallback_info: dict | None) -> tuple[int, int, int]:
    """매수 후보 강도 분포에 따라 이번 iter 의 cap 결정.

    Rules:
      - fallback_info != None (stale 데이터) → cap 확장 안 함 (over-trade 방지).
      - buy_cands 중 보유 안 한 종목에서 strength >= STRONG_THRESHOLD 인 갯수가
        free_slots 보다 많으면 그 초과분만큼 cap 확장. 단 MAX_CAP_EXTENSION 상한.

    Returns: (cap_effective, strong_count, extension)
      extension = 0 이면 평소 cap 그대로.
    """
    if fallback_info is not None or buy_cands.empty:
        return cfg_max_concurrent, 0, 0

    held = set(positions.keys())
    not_held = buy_cands[~buy_cands["symbol"].isin(held)]
    if not_held.empty:
        return cfg_max_concurrent, 0, 0

    strengths = not_held[["flow_ratio", "inst_ratio"]].min(axis=1)
    strong_count = int((strengths >= STRONG_THRESHOLD).sum())

    free_slots_normal = cfg_max_concurrent - len(positions)
    if strong_count <= max(0, free_slots_normal):
        return cfg_max_concurrent, strong_count, 0

    extension = min(strong_count - max(0, free_slots_normal), MAX_CAP_EXTENSION)
    return cfg_max_concurrent + extension, strong_count, extension


def _filter_pass(df_buy, df_sell, threshold_buy: float, threshold_sell: float,
                 whitelist: list, blacklist: list):
    """flow/inst ratio 임계 통과한 후보만 추출 — 매수/매도 임계 비대칭(E2)."""
    if not df_buy.empty:
        buy_pass = df_buy[
            (df_buy["flow_ratio"] > threshold_buy) & (df_buy["inst_ratio"] > threshold_buy)
        ].copy()
    else:
        buy_pass = df_buy.copy()
    if not df_sell.empty:
        sell_pass = df_sell[
            (df_sell["flow_ratio"] < -threshold_sell) & (df_sell["inst_ratio"] < -threshold_sell)
        ].copy()
    else:
        sell_pass = df_sell.copy()
    if whitelist or blacklist:
        buy_pass = apply_symbol_filter(buy_pass, whitelist or [], blacklist or [])
        sell_pass = apply_symbol_filter(sell_pass, [], blacklist or [])
    return buy_pass, sell_pass


def fetch_dual_candidates(client: KISClient, market_code: str = "0001",
                           whitelist: list | None = None,
                           blacklist: list | None = None,
                           cache_dir: Path | None = None):
    """매수 + 매도 상위 60 → dual 통과 후보 분리 반환.

    HTTP 500 등 일시 장애에 retry (1.5/3/6s backoff). 최종 실패 시 빈 DataFrame
    반환 (None 아님) — caller 는 빈 결과 = "신호 없음" 으로 안전하게 처리.

    Fallback (cache_dir 지정 시):
      - 가집계 API 가 매수+매도 둘 다 빈 결과 → disk cache 의 직전 good payload
        사용. 시초 09:00~09:15 KIS 모의서버 500 패턴을 메우기 위함.
      - Stale 데이터이므로 임계값을 FALLBACK_THRESHOLD_MULT (1.5x) 적용해
        false signal 위험 제한.
      - 48h 초과 stale 은 무시 (PersistentPayloadCache 가 처리).

    Returns: (buy_pass, sell_pass, fallback_info)
      fallback_info: None (정상) 또는
                     {"age_hours": float, "threshold": float} (fallback 사용)
    """
    import logging
    log = logging.getLogger(__name__)

    payload_buy = _safe_fetch_with_retry(
        lambda: client.foreign_institution_total(market_code=market_code, rank_sort="0"),
        label="가집계 매수상위",
    )
    time.sleep(1.5)
    payload_sell = _safe_fetch_with_retry(
        lambda: client.foreign_institution_total(market_code=market_code, rank_sort="1"),
        label="가집계 매도상위",
    )

    df_buy = compute_flow_ratios(parse_foreign_institution_response(payload_buy or {}))
    df_sell = compute_flow_ratios(parse_foreign_institution_response(payload_sell or {}))

    fallback_info = None
    primary_empty = df_buy.empty and df_sell.empty

    if not primary_empty and cache_dir is not None:
        try:
            PersistentPayloadCache(cache_dir=cache_dir).save(
                market_code, payload_buy, payload_sell,
            )
        except Exception as e:
            log.warning(f"  [캐시] save 실패: {e}")

    if primary_empty and cache_dir is not None:
        cached = PersistentPayloadCache(cache_dir=cache_dir).load(market_code)
        if cached is not None:
            cached_buy, cached_sell, age_h = cached
            df_buy = compute_flow_ratios(
                parse_foreign_institution_response(cached_buy)
            )
            df_sell = compute_flow_ratios(
                parse_foreign_institution_response(cached_sell)
            )
            stricter = ENTER_THRESHOLD * FALLBACK_THRESHOLD_MULT
            fallback_info = {"age_hours": age_h, "threshold": stricter}
            log.warning(
                f"  [FALLBACK] 가집계 API 빈 결과 — disk cache "
                f"({age_h:.1f}h 전) 사용, 임계값 {stricter:.3f} 강화 적용"
            )

    # 비대칭 임계: 매수 ENTER_THRESHOLD, 매도 SELL_THRESHOLD. fallback 시 둘 다 1.5배 강화.
    mult = FALLBACK_THRESHOLD_MULT if fallback_info is not None else 1.0
    threshold_buy = ENTER_THRESHOLD * mult
    threshold_sell = SELL_THRESHOLD * mult
    buy_pass, sell_pass = _filter_pass(
        df_buy, df_sell, threshold_buy, threshold_sell,
        whitelist or [], blacklist or [],
    )

    if primary_empty and fallback_info is None:
        log.warning("  [데이터] 가집계 API 매수+매도 둘 다 빈 결과 — 이번 iter 신호 없음 (매수/매도 모두 차단)")

    return buy_pass, sell_pass, fallback_info


def is_market_open(now: datetime | None = None) -> bool:
    """KOSPI 정규장 시간 (09:00 ~ 15:30 KST, 월~금)."""
    now = now or datetime.now()
    if now.weekday() >= 5:   # 토(5), 일(6)
        return False
    h, m = now.hour, now.minute
    after_open = (h, m) >= (9, 0)
    before_close = (h, m) <= (15, 30)
    return after_open and before_close


def run_one_iteration(client: KISClient, args, log):
    """한 사이클 — daemon 모드에서 매 interval 마다 호출."""
    state_path = Path(args.state)
    fresh = not state_path.exists()
    state = DualPaperState.load(state_path, initial_cash=args.initial_cash)
    if fresh:
        # Fresh start — 즉시 빈 state 저장 (dashboard 가 즉시 표시 가능)
        state.save(state_path)
        log.info(f"[iter] fresh start — state 파일 생성")

    # ===== 0. KIS 실잔고 sync (truth source) =====
    # 매 iter 시작 시 state.positions 와 KIS 실잔고가 일치하는지 확인. 어긋나면
    # KIS 기준으로 재정렬 (KIS 가 truth — 봇이 잘못 알고 있으면 잘못된 결정 함).
    # 5/12 의 sell-order-failed 누락 + 5/18 의 buy-order-failed (008700) 같은
    # silent drift 를 매 iter 마다 catch.
    kis_total_eval = None   # KIS 총평가금액(순자산) — MDD truth. sync 성공 시 채움.
    try:
        bal = client.balance()
        if isinstance(bal, dict) and bal.get("rt_cd") == "0":
            out1 = bal.get("output1") or []

            def _i(s):
                return int(str(s or "0").replace(",", "") or "0")

            def _f(s):
                return float(str(s or "0").replace(",", "") or "0")

            from datetime import timedelta
            today_iso = datetime.now().date().isoformat()
            yesterday_iso = (datetime.now().date() - timedelta(days=1)).isoformat()

            def _estimate_entry_date(h: dict) -> str:
                """KIS balance 의 thdt/bfdy 필드로 매수일 추정.
                둘 다 0 이면 그 이전 매수 — 정확 모름 → 'unknown' 표시."""
                thdt = _i(h.get("thdt_buyqty"))
                bfdy = _i(h.get("bfdy_buy_qty"))
                qty = _i(h.get("hldg_qty"))
                if thdt >= qty and thdt > 0:
                    return today_iso
                if thdt > 0:
                    return today_iso        # 오늘 추가 매수 (이전+오늘)
                if bfdy > 0:
                    return yesterday_iso
                return "unknown"            # 그 전 매수 — KIS balance 만으로는 모름

            kis_holdings = {
                h.get("pdno"): {"qty": _i(h.get("hldg_qty")),
                                "avg": _f(h.get("pchs_avg_pric")),
                                "entry_date": _estimate_entry_date(h)}
                for h in out1 if _i(h.get("hldg_qty")) > 0
            }
            kis_codes = set(kis_holdings.keys())
            state_codes = set(state.positions.keys())
            # state에는 있는데 KIS엔 없음 → 가짜 보유 (주문 실패로 인한 phantom)
            phantom = state_codes - kis_codes
            # KIS에는 있는데 state엔 없음 → 누락 (sell 실패 또는 다른 경로)
            missing = kis_codes - state_codes
            if phantom or missing:
                log.warning(
                    f"[sync] KIS drift 감지 — phantom(state만 있음): "
                    f"{sorted(phantom)}, missing(KIS만 있음): {sorted(missing)} — "
                    f"KIS 기준으로 재정렬"
                )
                # phantom 제거 (cash 환원 — 어차피 KIS 가 reject 했으니 실제로는 안 나갔음)
                for sym in phantom:
                    p = state.positions[sym]
                    state.cash += p.qty * p.avg_entry  # 가짜로 차감했던 금액 복구
                    del state.positions[sym]
                # missing 추가 (KIS 의 매입가/수량/매수일 추정 그대로 차용)
                for sym in missing:
                    h = kis_holdings[sym]
                    state.positions[sym] = Position(
                        qty=h["qty"], avg_entry=h["avg"],
                        entry_date=h["entry_date"],
                    )
                    # cash 는 차감 — KIS 가 실제로 돈 썼으니 우리 추정도 줄여야
                    state.cash -= h["qty"] * h["avg"]
                # 기존 보유도 entry_date 가 'today' 인데 KIS thdt/bfdy 가 둘 다 0 이면
                # 잘못 찍힌 것 (이전 sync 버그) — 'unknown' 으로 정정
                for sym in state_codes & kis_codes:
                    h = kis_holdings.get(sym)
                    if not h:
                        continue
                    est = h["entry_date"]
                    pos = state.positions[sym]
                    if pos.entry_date == today_iso and est == "unknown":
                        pos.entry_date = "unknown"
                state.save(state_path)
                log.info(f"[sync] 재정렬 완료 — positions={len(state.positions)}, "
                         f"cash={state.cash:,.0f}")
            # entry_date 정정 — drift 유무와 무관하게 매 sync 시 확인.
            # 'today' 로 잘못 찍힌 이전 매수 종목들을 KIS thdt/bfdy 기준으로
            # 'unknown' 또는 어제로 정정 (한 번만 변경되면 그 후로는 stable).
            fixed_dates = 0
            for sym in list(state.positions.keys()):
                h = kis_holdings.get(sym)
                if not h:
                    continue
                pos = state.positions[sym]
                est = h["entry_date"]
                # 오늘로 잘못 찍힌 것을 KIS truth 로 정정
                if pos.entry_date == today_iso and est != today_iso:
                    pos.entry_date = est
                    fixed_dates += 1
            if fixed_dates > 0:
                log.info(f"[sync] entry_date 정정 {fixed_dates}건 — KIS thdt/bfdy 기준")
                state.save(state_path)

            # ===== Cash sync (KIS truth) =====
            # 봇은 매수 0.1% 수수료 가정 (cost = qty*price*1.001), 매도 0.999 등
            # 일률 추정으로 cash 갱신. 실제 KIS vts 의 fill 슬리피지·수수료·세금과
            # 어긋나 6주간 누적 drift 1.27M+ 관찰됨 (2026-06-05 EOD reconcile).
            # positions sync 와 동일하게 KIS dnca_tot_amt 를 truth 로 인정,
            # tolerance 초과 시 정정. in-flight 주문 흡수용으로 10K 임계.
            out2 = bal.get("output2") or []
            if out2 and isinstance(out2[0], dict):
                o2 = out2[0]
                # ===== phantom margin / T+2 정산 왜곡 방어 (2026-06-17) =====
                # KIS VTS 는 매수 후 예수금(dnca_tot_amt)을 음수로 반환(phantom margin
                # debt) 하고, 매도 후에도 T+2 정산 전까지 옛 값을 줘서 dnca 를 그대로
                # 믿으면 cash 가 망가짐 → 가짜 폭락 → 잘못된 MDD 청산(06-05/11/12/17 재발).
                # KIS 의 총평가금액(tot_evlu_amt)은 phantom/정산을 net 한 진짜 순자산이므로
                # 이걸 truth 로 삼고, 순현금 = 총평가 - 보유평가 로 역산한다.
                kis_total_eval = _f(o2.get("tot_evlu_amt"))
                kis_hold_eval = sum(_f(h.get("evlu_amt")) for h in out1
                                    if _i(h.get("hldg_qty")) > 0)
                if kis_total_eval > 0:
                    kis_cash = kis_total_eval - kis_hold_eval        # 순현금 (phantom net)
                    src = f"총평가 {kis_total_eval:,.0f} - 보유평가 {kis_hold_eval:,.0f}"
                else:
                    kis_cash = _f(o2.get("dnca_tot_amt"))            # fallback (구버전)
                    src = "dnca_tot_amt (총평가 없음)"
                cash_diff = state.cash - kis_cash
                if abs(cash_diff) > 10_000:
                    log.warning(
                        f"[sync] cash drift {cash_diff:+,.0f}원 — "
                        f"봇 {state.cash:,.0f} → 순현금 {kis_cash:,.0f} ({src}) 정정"
                    )
                    state.cash = kis_cash
                    state.save(state_path)
        else:
            log.debug(f"[sync] balance 응답 비정상 — sync skip (rt_cd="
                      f"{bal.get('rt_cd') if isinstance(bal, dict) else '?'})")
    except Exception as e:
        log.warning(f"[sync] KIS balance 호출 실패 — sync skip: {type(e).__name__}: {str(e)[:80]}")

    log.info(f"[iter] state: cash={state.cash:,.0f}원, "
             f"positions={len(state.positions)}, peak={state.portfolio_peak:,.0f}원, "
             f"cooldown={state.cooldown_remaining}일")

    # ===== 1. 보유 종목 현재가 마크 =====
    portfolio_total = state.cash
    pos_values = {}
    for sym, pos in state.positions.items():
        try:
            q = client.quote(sym)
            price = int(str(q["output"]["stck_prpr"]).replace(",", ""))
            val = pos.qty * price
            pos_values[sym] = (price, val)
            portfolio_total += val
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"[{sym}] quote 실패: {e}")
            pos_values[sym] = (pos.avg_entry, pos.qty * pos.avg_entry)
            portfolio_total += pos.qty * pos.avg_entry

    # ===== 2. MDD Cap 체크 =====
    # 평가는 KIS 총평가금액(순자산)을 truth 로 — phantom margin/T+2 왜곡 차단.
    # 없으면 cash+holdings 재구성값(portfolio_total) fallback.
    mdd_total = kis_total_eval if (kis_total_eval and kis_total_eval > 0) else portfolio_total
    # 손상 peak 복구: phantom cascade 로 음수/0 이 된 경우 현재 순자산으로 리셋.
    if state.portfolio_peak <= 0:
        log.warning(f"[peak-fix] 손상된 peak({state.portfolio_peak:,.0f}) → {mdd_total:,.0f} 복구")
        state.portfolio_peak = mdd_total
        state.save(state_path)
    dd = (mdd_total - state.portfolio_peak) / state.portfolio_peak if state.portfolio_peak > 0 else 0
    # Sanity 가드: 순자산이 음수거나 한 iter 에 -50% 초과 급락 = phantom/정산왜곡 의심
    # → 강제청산 보류 + 경보 (실제 -50%+ 폭락이면 다음 iter 들에서 정상 MDD 가 잡음).
    if mdd_total <= 0 or dd <= -0.50:
        log.error(f"!!! [MDD-guard] 비정상 평가 (순자산 {mdd_total:,.0f}원, dd {dd*100:.1f}%) "
                  f"— phantom/정산왜곡 의심, 강제청산 보류 (peak {state.portfolio_peak:,.0f})")
    elif dd <= -MDD_CAP:
        log.warning(f"!!! MDD CAP 발동 ({dd*100:.2f}% < -{MDD_CAP*100:.0f}%) — 전종목 청산")
        for sym, pos in list(state.positions.items()):
            cur_px, _ = pos_values.get(sym, (pos.avg_entry, 0))
            # 매도 주문 결과 확인 — 실패하면 state 에서 안 지움 (다음 iter 재시도)
            order_ok = args.dry_only
            if not args.dry_only:
                status, _ = safe_order(client, sym, pos.qty, "sell", log)
                order_ok = (status == "ok")
            if order_ok:
                pnl_krw_val = pos.qty * (cur_px - pos.avg_entry)
                pnl_pct_val = ((cur_px - pos.avg_entry) / pos.avg_entry
                               if pos.avg_entry > 0 else 0.0)
                _append_trade(
                    state_path,
                    ts=datetime.now().isoformat(),
                    side="FORCE_SELL", code=sym,
                    name=str(sym),  # MDD 분기는 sell_cands 없음
                    qty=pos.qty, price=cur_px, avg_entry=pos.avg_entry,
                    pnl_pct=pnl_pct_val, pnl_krw=pnl_krw_val,
                    reason="mdd_cap",
                )
                state.cash += pos.qty * cur_px * 0.999
                del state.positions[sym]
            else:
                log.warning(f"    {sym} 강제청산 실패 — state 유지, 다음 iter 재시도")
        # Peak 를 청산 후 자본으로 리셋 — 안 그러면 다음 iter dd 가 여전히 ≤-cap
        # 으로 계산돼 MDD 분기가 무한 재발동함. 단 잔여 포지션이 있으면 peak 를
        # 현재 total 로 (= 청산 후 cash + 남은 포지션 평가) 잡음.
        leftover_pos_value = sum(
            pos.qty * pos_values.get(sym, (pos.avg_entry, 0))[0]
            for sym, pos in state.positions.items()
        )
        state.portfolio_peak = state.cash + leftover_pos_value
        state.cooldown_remaining = COOLDOWN_DAYS
        state.last_run = datetime.now().isoformat()
        state.run_count += 1
        state.save(state_path)

        # MDD 분기에서도 history append — 차트 끊김 방지
        try:
            import pandas as pd
            history_path = state_path.parent / "dual_history.parquet"
            new_total = state.cash + leftover_pos_value
            new_row = {
                "timestamp": datetime.now().isoformat(),
                "cash": state.cash,
                "n_positions": len(state.positions),
                "portfolio_total": new_total,
                "portfolio_peak": state.portfolio_peak,
                "drawdown": 0.0,
                "cooldown_remaining": state.cooldown_remaining,
                "n_buy_candidates": 0,
                "n_sell_candidates": 0,
                "mdd_event": True,
            }
            if history_path.exists():
                hist = pd.read_parquet(history_path)
                hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
                # 기존 datetime64[ns] 컬럼 + 새 row 의 isoformat str 충돌 방지
                hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
            else:
                hist = pd.DataFrame([new_row])
            hist.to_parquet(history_path)
        except Exception as he:
            log.warning(f"[history] MDD 분기 append 실패: {he}")
        return

    # ===== 3. Dual Confirmation 후보 (user config 적용) =====
    user_cfg = load_symbol_config().get("stock", {})
    whitelist = user_cfg.get("whitelist") or []
    blacklist = user_cfg.get("blacklist") or []
    cfg_market = user_cfg.get("market_code") or args.market_code
    try:
        buy_cands, sell_cands, fallback_info = fetch_dual_candidates(
            client, market_code=cfg_market,
            whitelist=whitelist, blacklist=blacklist,
            cache_dir=state_path.parent,
        )
    except Exception as e:
        log.error(f"가집계 호출 실패: {e}")
        return

    filter_info = ""
    if whitelist:
        filter_info += f" (whitelist {len(whitelist)}종)"
    if blacklist:
        filter_info += f" (blacklist {len(blacklist)}종)"
    if fallback_info is not None:
        filter_info += f" [FALLBACK {fallback_info['age_hours']:.1f}h]"
    log.info(f"  매수 후보 {len(buy_cands)}종 / 매도 후보 {len(sell_cands)}종{filter_info}")

    # ===== KOSPI 지수 조회 (매도 #5 + 매수 시 기록용) =====
    kospi_now = 0.0
    try:
        kospi_now = _parse_kospi_quote(client.index_price(KOSPI_INDEX_CODE))
    except Exception as e:
        log.warning(f"  코스피 지수 조회 실패: {type(e).__name__} — #5 underperform 룰 skip")

    # ===== 4. 매도 (보유 중 dual 매도 신호) =====
    for sym in list(state.positions.keys()):
        if (not sell_cands.empty
                and "symbol" in sell_cands.columns
                and sym in sell_cands["symbol"].values):
            pos = state.positions[sym]
            cur_px, _ = pos_values.get(sym, (pos.avg_entry, 0))
            log.info(f"  매도: {sym} {pos.qty}주 @ {cur_px:,}원")
            order_ok = args.dry_only   # dry_only 일 땐 주문 안 하지만 state 는 업데이트
            if not args.dry_only:
                status, _ = safe_order(client, sym, pos.qty, "sell", log)
                order_ok = (status == "ok")
            # 주문 성공 시만 state 업데이트 (실패 시 다음 iteration 에 재시도)
            if order_ok:
                # PnL 계산 — pos.avg_entry 가 진짜 매수가 (state 기준)
                pnl_krw_val = pos.qty * (cur_px - pos.avg_entry)
                pnl_pct_val = ((cur_px - pos.avg_entry) / pos.avg_entry
                               if pos.avg_entry > 0 else 0.0)
                _append_trade(
                    state_path,
                    ts=datetime.now().isoformat(),
                    side="SELL", code=sym,
                    name=str(sell_cands.loc[sell_cands["symbol"] == sym, "name"].values[0])
                         if "name" in sell_cands.columns
                            and (sell_cands["symbol"] == sym).any() else sym,
                    qty=pos.qty, price=cur_px, avg_entry=pos.avg_entry,
                    pnl_pct=pnl_pct_val, pnl_krw=pnl_krw_val,
                    reason="dual_sell",
                )
                state.cash += pos.qty * cur_px * 0.999
                del state.positions[sym]
            else:
                log.warning(f"    {sym} 매도 실패 — state 유지, 다음 iteration 재시도")

    # ===== 4b. 코스피 대비 underperform 강제 청산 (#5) =====
    if kospi_now > 0:
        for sym in list(state.positions.keys()):
            pos = state.positions[sym]
            cur_px, _ = pos_values.get(sym, (pos.avg_entry, 0))
            should_sell, rel = should_force_sell_underperform(
                pos, cur_px, kospi_now,
            )
            if not should_sell:
                continue
            log.warning(
                f"  [#5 underperform 청산] {sym} {pos.qty}주 @ {cur_px:,}원 "
                f"— 코스피 대비 relative {rel * 100:+.2f}% "
                f"(임계 -{UNDERPERFORM_THRESHOLD * 100:.0f}%)"
            )
            order_ok = args.dry_only
            if not args.dry_only:
                status, _ = safe_order(client, sym, pos.qty, "sell", log)
                order_ok = (status == "ok")
            if order_ok:
                pnl_krw_val = pos.qty * (cur_px - pos.avg_entry)
                pnl_pct_val = ((cur_px - pos.avg_entry) / pos.avg_entry
                               if pos.avg_entry > 0 else 0.0)
                _append_trade(
                    state_path,
                    ts=datetime.now().isoformat(),
                    side="SELL", code=sym, name=sym,
                    qty=pos.qty, price=cur_px, avg_entry=pos.avg_entry,
                    pnl_pct=pnl_pct_val, pnl_krw=pnl_krw_val,
                    reason="underperform_kospi",
                )
                state.cash += pos.qty * cur_px * 0.999
                del state.positions[sym]
            else:
                log.warning(f"    {sym} underperform 청산 실패 — 다음 iter 재시도")

    # ===== 4c. A3 매도 — 단기 가격 모멘텀 음전 + 손실 영역 =====
    # 가격만으로 작동 → top-30 가시성 무관, 모든 보유 종목에 적용.
    # 조건 (둘 다 AND):
    #   ① 보유 종목의 10일 가격 수익률 < -5%  (= 지속적 하락 검증)
    #   ② 현재가 < 진입가  (= 손실 영역, 승자 보호 게이트)
    # 30위 밖 가시성 사각지대 종목의 매도 timing backup.
    # 백테스트(2026-06-05): Calmar 0.85→1.02, MDD -45%→-38%, bear +20%→+28%.
    for sym in list(state.positions.keys()):
        pos = state.positions[sym]
        cur_px, _ = pos_values.get(sym, (pos.avg_entry, 0))
        if pos.avg_entry <= 0 or cur_px <= 0:
            continue
        # 손실 영역 게이트 (승자 절대 안 자름)
        if cur_px >= pos.avg_entry:
            continue
        short_ret = _get_momentum(sym, lookback=SHORT_MOM_LOOKBACK)
        if short_ret is None:
            continue  # 가격 데이터 없으면 보수적으로 skip
        if short_ret >= SHORT_MOM_THRESHOLD:
            continue
        log.warning(
            f"  [A3 매도] {sym} {pos.qty}주 @ {cur_px:,}원 — "
            f"{SHORT_MOM_LOOKBACK}일 수익률 {short_ret*100:+.2f}% "
            f"(< {SHORT_MOM_THRESHOLD*100:.0f}%) + 손실영역"
        )
        order_ok = args.dry_only
        if not args.dry_only:
            status, _ = safe_order(client, sym, pos.qty, "sell", log)
            order_ok = (status == "ok")
        if order_ok:
            pnl_krw_val = pos.qty * (cur_px - pos.avg_entry)
            pnl_pct_val = (cur_px - pos.avg_entry) / pos.avg_entry
            _append_trade(
                state_path,
                ts=datetime.now().isoformat(),
                side="SELL", code=sym, name=sym,
                qty=pos.qty, price=cur_px, avg_entry=pos.avg_entry,
                pnl_pct=pnl_pct_val, pnl_krw=pnl_krw_val,
                reason="a3_short_momentum",
            )
            state.cash += pos.qty * cur_px * 0.999
            del state.positions[sym]
        else:
            log.warning(f"    {sym} A3 매도 실패 — 다음 iter 재시도")

    # ===== 5. 매수 (cooldown / circuit breaker 아니면) =====
    # config/symbols.json 의 max_concurrent 가 있으면 그 값 우선 (런타임 변경 즉시 반영)
    cfg_max_concurrent = int(user_cfg.get("max_concurrent") or MAX_CONCURRENT)
    cap_effective, strong_count, extension = compute_effective_cap(
        buy_cands, state.positions, cfg_max_concurrent, fallback_info,
    )
    if extension > 0:
        log.info(f"  [CAP 확장] strong 후보 {strong_count}종 — "
                 f"cap {cfg_max_concurrent} → {cap_effective} (이번 iter 한정)")
    if _is_halted(log):
        log.info(f"  circuit halt — 신규 매수 차단")
    elif state.cooldown_remaining > 0:
        log.info(f"  cooldown {state.cooldown_remaining}일 — 매수 차단")
    elif buy_cands.empty:
        log.info("  매수 후보 없음")
    else:
        # ===== 5a. Strength sort + 모멘텀60 confluence 필터 =====
        # WF 검증: 외인+기관 매수 AND 60일 가격 추세 양수일 때만 진입 → falling-knife 차단.
        buy_cands_sorted = buy_cands.copy()
        buy_cands_sorted["strength"] = buy_cands_sorted[["flow_ratio", "inst_ratio"]].min(axis=1)
        buy_cands_sorted = buy_cands_sorted.sort_values("strength", ascending=False)

        n_pre_mom = len(buy_cands_sorted)
        moms = {s: _get_momentum(s, lookback=MOMENTUM_LOOKBACK)
                for s in buy_cands_sorted["symbol"].tolist()}
        buy_cands_sorted = buy_cands_sorted[
            buy_cands_sorted["symbol"].apply(
                lambda s: moms.get(s) is not None and moms.get(s) > MOMENTUM_MIN
            )
        ].copy()
        n_no_data = sum(1 for v in moms.values() if v is None)
        n_neg = sum(1 for v in moms.values() if v is not None and v <= MOMENTUM_MIN)
        if n_pre_mom > 0:
            log.info(f"  [모멘텀{MOMENTUM_LOOKBACK}일] 통과 {len(buy_cands_sorted)}/{n_pre_mom}종 "
                     f"(데이터없음 {n_no_data}, 추세음수 {n_neg})")

        # ===== 5a-2. 매수 단기추세 가드 (Fix 1, 2026-06-09) =====
        # 60일은 +라 통과해도, 10일 단기 수익률이 BUY_SHORT_MOM_THRESHOLD 미만이면
        # A3 매도가 즉시 뱉어낼 falling-knife → 매수 보류 (churn 차단). 데이터없음(None)도 보류.
        if not buy_cands_sorted.empty:
            n_pre_guard = len(buy_cands_sorted)
            short_moms = {s: _get_momentum(s, lookback=SHORT_MOM_LOOKBACK)
                          for s in buy_cands_sorted["symbol"].tolist()}
            buy_cands_sorted = buy_cands_sorted[
                buy_cands_sorted["symbol"].apply(
                    lambda s: short_moms.get(s) is not None
                    and short_moms.get(s) >= BUY_SHORT_MOM_THRESHOLD
                )
            ].copy()
            n_guard_blocked = n_pre_guard - len(buy_cands_sorted)
            if n_guard_blocked > 0:
                log.info(f"  [매수가드{SHORT_MOM_LOOKBACK}일] {n_guard_blocked}종 차단 "
                         f"(10일 < {BUY_SHORT_MOM_THRESHOLD*100:.0f}% — A3 즉시매도 회피)")

        free_slots = cap_effective - len(state.positions)

        # ===== 5b. Replacement: 슬롯 가득 + 모멘텀 통과 후보 있으면 약자 교체 =====
        # 적격 = 보유 ≥ REPLACEMENT_MIN_HOLD 일 AND 손익 < REPLACEMENT_PNL_MAX.
        # 가장 손익 낮은 종목부터 다음날 시가에 청산해 자리 확보.
        if not buy_cands_sorted.empty and free_slots <= 0 and state.positions:
            today_dt = datetime.now()
            eligible = []
            for held_sym, held_pos in state.positions.items():
                days_held = _days_between(held_pos.entry_date, today_dt)
                if days_held < REPLACEMENT_MIN_HOLD:
                    continue
                if held_pos.avg_entry <= 0:
                    continue
                cur_px, _ = pos_values.get(held_sym, (held_pos.avg_entry, 0))
                pnl = (cur_px - held_pos.avg_entry) / held_pos.avg_entry
                if pnl < REPLACEMENT_PNL_MAX:
                    eligible.append((held_sym, pnl, cur_px, held_pos, days_held))
            eligible.sort(key=lambda x: x[1])  # 손익 낮은 순

            n_excess = len(buy_cands_sorted)
            n_evict = min(n_excess, len(eligible))
            if n_evict > 0:
                log.info(f"  [REPLACEMENT] 슬롯 cap 도달 + 모멘텀 통과 후보 {n_excess}종 — "
                         f"적격 약자 {len(eligible)}종 중 {n_evict}종 교체")
                for k in range(n_evict):
                    ev_sym, ev_pnl, ev_px, ev_pos, ev_days = eligible[k]
                    log.info(f"    교체매도: {ev_sym} {ev_pos.qty}주 @ {ev_px:,}원 "
                             f"(보유 {ev_days}일, PnL {ev_pnl*100:+.2f}%)")
                    order_ok = args.dry_only
                    if not args.dry_only:
                        status, _ = safe_order(client, ev_sym, ev_pos.qty, "sell", log)
                        order_ok = (status == "ok")
                    if order_ok:
                        pnl_krw_val = ev_pos.qty * (ev_px - ev_pos.avg_entry)
                        _append_trade(
                            state_path, ts=datetime.now().isoformat(),
                            side="SELL", code=ev_sym, name=ev_sym,
                            qty=ev_pos.qty, price=ev_px, avg_entry=ev_pos.avg_entry,
                            pnl_pct=ev_pnl, pnl_krw=pnl_krw_val,
                            reason="replacement",
                        )
                        state.cash += ev_pos.qty * ev_px * 0.999
                        del state.positions[ev_sym]
                        free_slots += 1
                    else:
                        log.warning(f"    {ev_sym} 교체매도 실패 — 다음 iter 재시도")
            else:
                log.info(f"  [REPLACEMENT] 후보 {n_excess}종 있으나 적격 약자 없음 "
                         f"(모두 보유 {REPLACEMENT_MIN_HOLD}일 미만이거나 수익 ≥{REPLACEMENT_PNL_MAX*100:.0f}%)")

        # ===== 5c. 매수 실행 =====
        if buy_cands_sorted.empty:
            log.info("  모멘텀 필터 통과 0종 — 매수 없음")
        elif free_slots <= 0:
            log.info(f"  포지션 cap ({cap_effective}) 도달 — 추가 매수 안 함")
        else:
            n_to_buy = min(len(buy_cands_sorted), free_slots)
            divisor = max(n_to_buy, MIN_SIZING_DIVISOR)
            slot_cash = state.cash / divisor

            buys = 0
            for _, row in buy_cands_sorted.iterrows():
                if buys >= free_slots:
                    break
                sym = row["symbol"]
                if sym in state.positions:
                    continue
                price = int(row["price"])
                if price <= 0:
                    continue
                volume = int(row.get("volume", 0))

                # ===== 유동성 필터 (Phase 1 + 2) =====
                asking = None
                try:
                    asking = client.asking_price(sym)
                    time.sleep(0.5)
                except Exception as e:
                    log.warning(f"  [{sym}] asking_price 호출 실패: {type(e).__name__} — Phase 1 만 적용")

                check = evaluate_liquidity(
                    sym, price=price, acml_vol=volume,
                    asking_payload=asking,
                    min_trading_value_krw=MIN_TRADING_VALUE,
                    max_spread_bps=MAX_SPREAD,
                )
                if not check.passed:
                    log.info(f"  [skip] {sym} {row.get('name','')}: {check.reason}")
                    continue

                qty = int(slot_cash // (price * 1.001))
                if qty <= 0:
                    log.info(f"  [skip] {sym}: slot 자본 부족 ({slot_cash:,.0f} vs {price:,})")
                    continue
                cost = qty * price * 1.001
                bps_str = f", spread {check.spread_bps:.1f}bps" if check.spread_bps is not None else ""
                log.info(f"  매수: {sym} {row.get('name','')} {qty}주 @ {price:,}원 "
                         f"(flow {row['flow_ratio']*100:+.2f}% / inst {row['inst_ratio']*100:+.2f}%, "
                         f"거래대금 {check.trading_value/1e8:.1f}억{bps_str})")
                # KIS 주문이 OK 한 경우만 state 갱신. dry_only 면 무조건 OK.
                order_ok = args.dry_only
                if not args.dry_only:
                    status, _ = safe_order(client, sym, qty, "buy", log)
                    order_ok = (status == "ok")
                if order_ok:
                    _append_trade(
                        state_path,
                        ts=datetime.now().isoformat(),
                        side="BUY", code=sym, name=str(row.get("name", "")) or sym,
                        qty=qty, price=float(price),
                        avg_entry=float(price), pnl_pct=None, pnl_krw=None,
                        flow=float(row.get("flow_ratio", 0)) * 100,
                        inst=float(row.get("inst_ratio", 0)) * 100,
                        reason="dual_buy",
                    )
                    state.cash -= cost
                    state.positions[sym] = Position(
                        qty=qty, avg_entry=float(price),
                        entry_date=datetime.now().date().isoformat(),
                        kospi_at_entry=kospi_now if kospi_now > 0 else None,
                    )
                    buys += 1
                else:
                    log.warning(f"    {sym} 매수 실패 — state 변경 안 함 (KIS 실주문 X)")

    # ===== 6. portfolio + history =====
    new_total = state.cash + sum(
        pos.qty * pos_values.get(sym, (pos.avg_entry, 0))[0]
        for sym, pos in state.positions.items()
    )
    if new_total > state.portfolio_peak:
        state.portfolio_peak = new_total

    state.last_run = datetime.now().isoformat()
    state.run_count += 1
    state.save(state_path)

    # heartbeat — dashboard 가 mtime 으로 봇 alive 판단
    try:
        import os as _os
        hb = state_path.parent / "dual_heartbeat.txt"
        hb.write_text(
            f"{datetime.now().isoformat()}\n"
            f"pid={_os.getpid()}\n"
            f"iter={state.run_count}\n"
            f"positions={len(state.positions)}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    history_path = state_path.parent / "dual_history.parquet"
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "cash": state.cash,
        "n_positions": len(state.positions),
        "portfolio_total": new_total,
        "portfolio_peak": state.portfolio_peak,
        "drawdown": (new_total - state.portfolio_peak) / state.portfolio_peak if state.portfolio_peak > 0 else 0,
        "cooldown_remaining": state.cooldown_remaining,
        "n_buy_candidates": len(buy_cands),
        "n_sell_candidates": len(sell_cands),
    }
    try:
        import pandas as pd
        if history_path.exists():
            hist = pd.read_parquet(history_path)
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
            # 기존 datetime64[ns] 컬럼 + 새 row 의 isoformat str 충돌 방지 (2026-06-05 fix)
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        else:
            hist = pd.DataFrame([new_row])
        hist.to_parquet(history_path)
    except Exception as e:
        log.warning(f"history append failed: {e}")

    log.info(f"  → portfolio {new_total:,.0f}원 ({(new_total/state.initial_cash-1)*100:+.2f}%), "
             f"보유 {len(state.positions)}종")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(ROOT / "data" / "dual_state.json"))
    ap.add_argument("--log", default=str(ROOT / "data" / "dual_trading.log"))
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--market-code", default="0001",
                    choices=["0001", "1001"], help="0001 KOSPI / 1001 KOSDAQ")
    ap.add_argument("--dry-only", action="store_true",
                    help="실제 주문 안 함 — 신호 + 가상 포지션 갱신만 (Phase 1)")
    ap.add_argument("--watch", action="store_true",
                    help="daemon 모드 — interval 마다 자동 polling (장중에만)")
    ap.add_argument("--interval", type=int, default=300,
                    help="watch 간격 (초). default 300 (5분)")
    ap.add_argument("--off-hours-skip", action="store_true",
                    help="watch 모드에서 장외 시간엔 polling 건너뜀")
    args = ap.parse_args()

    setup_logging(Path(args.log))
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info(f"Dual Paper Trading @ {datetime.now().isoformat()}")
    log.info(f"market={args.market_code}, dry_only={args.dry_only}, "
             f"watch={args.watch}, interval={args.interval}s")
    log.info(f"params: threshold={ENTER_THRESHOLD}, max_concurrent={MAX_CONCURRENT}, "
             f"mdd_cap={MDD_CAP}, cooldown={COOLDOWN_DAYS}, "
             f"min_sizing_divisor={MIN_SIZING_DIVISOR}")
    log.info("=" * 60)

    load_dotenv(ROOT / ".env")
    cfg = KISConfig.from_env()
    if not cfg.app_key:
        log.error("KIS_APP_KEY 미설정 — 종료")
        return

    client = KISClient(cfg)
    try:
        client.token()
    except Exception as e:
        log.error(f"KIS token 발급 실패: {e}")
        return

    log.info(f"KIS env={cfg.env}, dry_run={cfg.dry_run}, account={cfg.account_no[:4]}****")

    # ===== Watch (daemon) 모드 =====
    if args.watch:
        log.info(f"\n=== Watch 모드 시작 — {args.interval}초마다 polling ===")
        log.info("(Ctrl+C 로 종료)")
        iter_count = 0
        try:
            while True:
                iter_count += 1
                log.info(f"\n--- iter #{iter_count} @ {datetime.now().strftime('%H:%M:%S')} ---")
                if args.off_hours_skip and not is_market_open():
                    log.info("  장외 시간 — polling 건너뜀")
                else:
                    try:
                        run_one_iteration(client, args, log)
                    except Exception as e:
                        log.exception(f"iteration error: {e}")
                # watch loop heartbeat — run_one_iteration 안 실패하거나 장외라도 봇 alive 표시
                try:
                    import os as _os
                    hb = Path(args.state).parent / "dual_heartbeat.txt"
                    hb.write_text(
                        f"{datetime.now().isoformat()}\n"
                        f"pid={_os.getpid()}\n"
                        f"iter={iter_count}\n"
                        f"interval={args.interval}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                log.info(f"  ⏰ 다음 polling 까지 {args.interval}s 대기")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("\n=== Watch 종료 (Ctrl+C) ===")
        return

    # ===== One-shot 모드 (기존) =====

    # State 로드
    state_path = Path(args.state)
    state = DualPaperState.load(state_path, initial_cash=args.initial_cash)
    log.info(f"state loaded: cash={state.cash:,.0f}원, "
             f"positions={len(state.positions)}, "
             f"peak={state.portfolio_peak:,.0f}원, "
             f"cooldown={state.cooldown_remaining}일")

    # ===== 1. 보유 종목 현재가 마크 + portfolio total =====
    portfolio_total = state.cash
    pos_values = {}
    for sym, pos in state.positions.items():
        try:
            q = client.quote(sym)
            price = int(str(q["output"]["stck_prpr"]).replace(",", ""))
            val = pos.qty * price
            pos_values[sym] = (price, val)
            portfolio_total += val
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"[{sym}] quote 실패: {e}")
            pos_values[sym] = (0, pos.qty * pos.avg_entry)
            portfolio_total += pos.qty * pos.avg_entry

    log.info(f"\n=== Portfolio Mark ===")
    log.info(f"  cash       : {state.cash:>14,.0f}원")
    for sym, pos in state.positions.items():
        cur_px, val = pos_values.get(sym, (0, 0))
        pnl_pct = (cur_px / pos.avg_entry - 1) * 100 if pos.avg_entry > 0 else 0
        log.info(f"  {sym} : {pos.qty}주 × {cur_px:,}원 = {val:>14,.0f}원  ({pnl_pct:+.2f}%)")
    log.info(f"  TOTAL      : {portfolio_total:>14,.0f}원  "
             f"(peak {state.portfolio_peak:,.0f}원, "
             f"DD {(portfolio_total - state.portfolio_peak) / state.portfolio_peak * 100:+.2f}%)")

    # ===== 2. MDD Cap 체크 =====
    dd = (portfolio_total - state.portfolio_peak) / state.portfolio_peak if state.portfolio_peak > 0 else 0
    mdd_triggered = dd <= -MDD_CAP

    if mdd_triggered:
        log.warning(f"\n!!! MDD CAP 발동 ({dd*100:.2f}% < -{MDD_CAP*100:.0f}%) — 전종목 청산")
        for sym, pos in list(state.positions.items()):
            cur_px, _ = pos_values.get(sym, (pos.avg_entry, 0))
            log.info(f"  강제 청산: {sym} {pos.qty}주 @ {cur_px:,}원")
            if not args.dry_only:
                try:
                    client.order(sym, qty=pos.qty, side="sell", price=0, ord_div="01")  # 시장가
                except Exception as e:
                    log.error(f"    sell order failed: {e}")
            state.cash += pos.qty * cur_px * 0.999  # 비용 가정
            del state.positions[sym]
        # Peak 를 청산 후 자본으로 리셋 — watch 모드와 동일 (MDD 무한 재발동 방지).
        state.portfolio_peak = state.cash
        state.cooldown_remaining = COOLDOWN_DAYS
        # 갱신 후 즉시 종료
        state.last_run = datetime.now().isoformat()
        state.run_count += 1
        state.save(state_path)
        log.info("MDD cap triggered — 종료")
        return

    # ===== 3. Dual Confirmation 후보 식별 =====
    log.info(f"\n=== Dual Confirmation Scan ===")
    try:
        buy_cands, sell_cands, fallback_info = fetch_dual_candidates(
            client, market_code=args.market_code, cache_dir=state_path.parent,
        )
    except Exception as e:
        log.error(f"가집계 호출 실패: {e}")
        return

    if fallback_info is not None:
        log.warning(f"[FALLBACK 적용] disk cache {fallback_info['age_hours']:.1f}h "
                    f"전 데이터 사용, 임계값 {fallback_info['threshold']:.3f}")
    log.info(f"매수 후보 ({len(buy_cands)}종): "
             f"{', '.join(buy_cands['symbol'].tolist()) if not buy_cands.empty else '(없음)'}")
    log.info(f"매도 후보 ({len(sell_cands)}종): "
             f"{', '.join(sell_cands['symbol'].tolist()) if not sell_cands.empty else '(없음)'}")

    # ===== KOSPI 지수 조회 (#5 underperform + 매수 시 기록용) =====
    kospi_now = 0.0
    try:
        kospi_now = _parse_kospi_quote(client.index_price(KOSPI_INDEX_CODE))
    except Exception as e:
        log.warning(f"  코스피 지수 조회 실패: {type(e).__name__} — #5 underperform 룰 skip")

    # ===== 4. 매도 (보유 중 dual 매도 신호) =====
    log.info(f"\n=== 매도 결정 ===")
    sells_executed = 0
    for sym in list(state.positions.keys()):
        if (not sell_cands.empty
                and "symbol" in sell_cands.columns
                and sym in sell_cands["symbol"].values):
            pos = state.positions[sym]
            cur_px, _ = pos_values.get(sym, (pos.avg_entry, 0))
            log.info(f"  매도: {sym} {pos.qty}주 @ {cur_px:,}원")
            if not args.dry_only:
                try:
                    client.order(sym, qty=pos.qty, side="sell", price=0, ord_div="01")
                except Exception as e:
                    log.error(f"    sell order failed: {e}")
            state.cash += pos.qty * cur_px * 0.999
            del state.positions[sym]
            sells_executed += 1
    if sells_executed == 0:
        log.info("  (매도 신호 없음)")

    # ===== 4b. 코스피 대비 underperform 강제 청산 (#5) =====
    if kospi_now > 0:
        underperform_sells = 0
        for sym in list(state.positions.keys()):
            pos = state.positions[sym]
            cur_px, _ = pos_values.get(sym, (pos.avg_entry, 0))
            should_sell, rel = should_force_sell_underperform(
                pos, cur_px, kospi_now,
            )
            if not should_sell:
                continue
            log.warning(
                f"  [#5 underperform 청산] {sym} {pos.qty}주 — "
                f"relative {rel * 100:+.2f}%"
            )
            if not args.dry_only:
                try:
                    client.order(sym, qty=pos.qty, side="sell", price=0, ord_div="01")
                except Exception as e:
                    log.error(f"    underperform sell order failed: {e}")
            state.cash += pos.qty * cur_px * 0.999
            del state.positions[sym]
            underperform_sells += 1
        if underperform_sells > 0:
            log.info(f"  (underperform 청산 {underperform_sells}종)")

    # ===== 5. 매수 (cooldown 아니면) =====
    log.info(f"\n=== 매수 결정 ===")
    cap_effective, strong_count, extension = compute_effective_cap(
        buy_cands, state.positions, MAX_CONCURRENT, fallback_info,
    )
    if extension > 0:
        log.info(f"  [CAP 확장] strong 후보 {strong_count}종 — "
                 f"cap {MAX_CONCURRENT} → {cap_effective}")
    if state.cooldown_remaining > 0:
        log.info(f"  cooldown {state.cooldown_remaining}일 남음 — 매수 차단")
        state.cooldown_remaining -= 1
    else:
        free_slots = cap_effective - len(state.positions)
        if free_slots <= 0:
            log.info(f"  포지션 cap ({cap_effective}) 도달 — 추가 매수 안 함")
        elif buy_cands.empty:
            log.info("  매수 후보 없음")
        else:
            # 강도 (min(flow, inst)) 내림차순 우선
            buy_cands_sorted = buy_cands.copy()
            buy_cands_sorted["strength"] = buy_cands_sorted[["flow_ratio", "inst_ratio"]].min(axis=1)
            buy_cands_sorted = buy_cands_sorted.sort_values("strength", ascending=False)

            # Cash-aware sizing: 매수할 후보 수에 맞춰 자본 분산
            # 최소 MIN_SIZING_DIVISOR(3) 으로 한 종목 올인 방지
            n_to_buy = min(len(buy_cands_sorted), free_slots)
            divisor = max(n_to_buy, MIN_SIZING_DIVISOR)
            slot_cash = state.cash / divisor
            log.info(f"  free slots {free_slots}, 후보 {len(buy_cands_sorted)}, "
                     f"실매수 {n_to_buy} → divisor {divisor}, slot 자본 ~{slot_cash:,.0f}원")

            buys_executed = 0
            for _, row in buy_cands_sorted.iterrows():
                if buys_executed >= free_slots:
                    break
                sym = row["symbol"]
                if sym in state.positions:
                    continue
                price = int(row["price"])
                if price <= 0:
                    continue
                volume = int(row.get("volume", 0))

                # ===== 유동성 필터 (Phase 1 + 2) =====
                asking = None
                try:
                    asking = client.asking_price(sym)
                    time.sleep(0.5)
                except Exception as e:
                    log.warning(f"  [{sym}] asking_price 호출 실패: {type(e).__name__}")

                check = evaluate_liquidity(
                    sym, price=price, acml_vol=volume,
                    asking_payload=asking,
                    min_trading_value_krw=MIN_TRADING_VALUE,
                    max_spread_bps=MAX_SPREAD,
                )
                if not check.passed:
                    log.info(f"  [skip] {sym} {row.get('name','')}: {check.reason}")
                    continue

                qty = int(slot_cash // (price * 1.001))
                if qty <= 0:
                    log.info(f"  [skip] {sym}: slot 자본 부족 ({slot_cash:,.0f} vs {price:,})")
                    continue
                cost = qty * price * 1.001
                bps_str = f", spread {check.spread_bps:.1f}bps" if check.spread_bps is not None else ""
                log.info(f"  매수: {sym} {row['name']} {qty}주 @ {price:,}원 "
                         f"(flow {row['flow_ratio']*100:+.2f}% / inst {row['inst_ratio']*100:+.2f}%, "
                         f"거래대금 {check.trading_value/1e8:.1f}억{bps_str})")
                if not args.dry_only:
                    try:
                        client.order(sym, qty=qty, side="buy", price=0, ord_div="01")
                    except Exception as e:
                        log.error(f"    buy order failed: {e}")
                state.cash -= cost
                state.positions[sym] = Position(
                    qty=qty, avg_entry=float(price),
                    entry_date=datetime.now().date().isoformat(),
                    kospi_at_entry=kospi_now if kospi_now > 0 else None,
                )
                buys_executed += 1
            if buys_executed == 0:
                log.info("  (실제 매수 0건)")

    # ===== 6. 새 portfolio total + peak 갱신 =====
    new_total = state.cash + sum(
        pos.qty * pos_values.get(sym, (pos.avg_entry, 0))[0]
        for sym, pos in state.positions.items()
    )
    if new_total > state.portfolio_peak:
        state.portfolio_peak = new_total
        log.info(f"\n신고점 갱신: {new_total:,.0f}원")

    state.last_run = datetime.now().isoformat()
    state.run_count += 1
    state.save(state_path)

    # Append-only daily history (대시보드 차트용)
    history_path = state_path.parent / "dual_history.parquet"
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "cash": state.cash,
        "n_positions": len(state.positions),
        "portfolio_total": new_total,
        "portfolio_peak": state.portfolio_peak,
        "drawdown": (new_total - state.portfolio_peak) / state.portfolio_peak if state.portfolio_peak > 0 else 0,
        "cooldown_remaining": state.cooldown_remaining,
        "n_buy_candidates": len(buy_cands) if 'buy_cands' in dir() else 0,
        "n_sell_candidates": len(sell_cands) if 'sell_cands' in dir() else 0,
    }
    try:
        import pandas as pd
        if history_path.exists():
            hist = pd.read_parquet(history_path)
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
            # 기존 datetime64[ns] 컬럼 + 새 row 의 isoformat str 충돌 방지 (2026-06-05 fix)
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
        else:
            hist = pd.DataFrame([new_row])
        hist.to_parquet(history_path)
        log.info(f"  history append → {history_path}")
    except Exception as e:
        log.warning(f"history append failed: {e}")

    log.info(f"\n=== 종료 ===")
    log.info(f"  최종 cash      : {state.cash:>14,.0f}원")
    log.info(f"  최종 portfolio : {new_total:>14,.0f}원 ({(new_total/state.initial_cash-1)*100:+.2f}%)")
    log.info(f"  보유 종목 수   : {len(state.positions)}")
    log.info(f"  state file     : {state_path}")


if __name__ == "__main__":
    main()
