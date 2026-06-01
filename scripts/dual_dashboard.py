"""Dual Confirmation Paper Trading 대시보드 (Streamlit).

매 5초 자동 새로고침. dual_state.json + dual_history.parquet 시각화.

Usage:
  pip install streamlit plotly
  streamlit run scripts/dual_dashboard.py

표시:
  - 현재 자본 / 누적 수익률 / drawdown / cooldown 상태
  - 보유 종목 (qty, avg_entry, 현재가, 평가금액, P&L)
  - 오늘 신호 후보 (매수 / 매도)
  - portfolio total 시계열 차트
  - 매수 후보 vs 매도 후보 시계열
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REFRESH_SEC = 30   # 너무 자주 갱신하면 KIS API quota 문제

# load env + KIS
sys.path.insert(0, str(ROOT / "packages" / "autotrader" / "src"))
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from autotrader.broker.kis_client import KISClient, KISConfig
from autotrader.market.foreign_inst_realtime import (
    parse_foreign_institution_response, compute_flow_ratios,
)


STATE_PATH = ROOT / "data" / "dual_state.json"
HISTORY_PATH = ROOT / "data" / "dual_history.parquet"
TRADES_PATH = ROOT / "data" / "dual_trades.parquet"
TRADING_LOG = ROOT / "data" / "dual_trading.log"
HEARTBEAT_PATH = ROOT / "data" / "dual_heartbeat.txt"
SYMBOL_NAMES_CACHE = ROOT / "data" / "symbol_names.json"
CIRCUIT_PATH = ROOT / "data" / "dual_circuit.json"

# 종목 이름 매핑 (KIS quote 응답에 이름 비어있을 때 fallback + 가독성)
SYMBOL_NAMES = {
    "005930": "삼성전자",      "000660": "SK하이닉스",   "035420": "NAVER",
    "035720": "카카오",         "005380": "현대차",       "051910": "LG화학",
    "005490": "POSCO홀딩스",   "207940": "삼성바이오로직스", "105560": "KB금융",
    "055550": "신한지주",       "066570": "LG전자",       "068270": "셀트리온",
    "006400": "삼성SDI",        "028260": "삼성물산",     "003670": "포스코퓨처엠",
    "000270": "기아",           "012330": "현대모비스",   "010130": "고려아연",
    "011170": "롯데케미칼",    "086790": "하나금융지주", "024110": "기업은행",
    "003490": "대한항공",       "042660": "한화오션",     "042700": "한미반도체",
    "034020": "두산에너빌리티", "009830": "한화솔루션",   "011200": "HMM",
    "011780": "금호석유",       "086280": "현대글로비스", "047810": "한국항공우주",
    "015760": "한국전력",       "030200": "KT",          "003550": "LG",
    "004020": "현대제철",       "047040": "대우건설",     "138930": "BNK금융지주",
    "005935": "삼성전자우",     "326030": "SK바이오팜",   "395400": "SK리츠",
    "000720": "현대건설",       "007340": "DN오토모티브",
    "034220": "LG디스플레이",   "073240": "금호타이어",   "088980": "맥쿼리인프라",
}


def _load_name_cache() -> dict:
    if not SYMBOL_NAMES_CACHE.exists():
        return {}
    try:
        return json.loads(SYMBOL_NAMES_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_name_cache(cache: dict) -> None:
    try:
        SYMBOL_NAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SYMBOL_NAMES_CACHE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


@st.cache_resource
def _name_cache_singleton() -> dict:
    """Streamlit 세션 lifetime cache (파일에서 1회 load, 이후 in-memory)."""
    return _load_name_cache()


@st.cache_data(ttl=300)
def _investor_trend_for(sym: str) -> dict | None:
    """30위 밖 보유 종목의 외인/기관 순매수 비율 fallback 조회.

    KIS investor_trend (FHKST01010900) 응답은 acml_vol 필드가 없지만
    개인/외인/기관 각 주체의 매수 수량(*_shnu_vol)과 매도 수량(*_seln_vol)을
    모두 포함한다. 매수합 ≈ 매도합 ≈ 일거래량 이므로 그 평균으로 분모를
    재구성해 가집계와 동일한 의미의 비율을 계산한다.
    """
    client = get_kis_client()
    if client is None:
        return None
    try:
        r = client.investor_trend(sym)
        if r.get("rt_cd") != "0":
            return None
        out = r.get("output") or []
        if not out:
            return None
        latest = out[0]
        def i(s):
            return int(str(s or "0").replace(",", ""))
        buy_total = i(latest.get("prsn_shnu_vol")) + i(latest.get("frgn_shnu_vol")) + i(latest.get("orgn_shnu_vol"))
        sell_total = i(latest.get("prsn_seln_vol")) + i(latest.get("frgn_seln_vol")) + i(latest.get("orgn_seln_vol"))
        vol = (buy_total + sell_total) // 2
        if vol <= 0:
            return None
        return {
            "flow_ratio": i(latest.get("frgn_ntby_qty")) / vol,
            "inst_ratio": i(latest.get("orgn_ntby_qty")) / vol,
            "src": "investor_trend (전영업일)",
        }
    except Exception:
        return None


def _resolve_name_via_kis(sym: str) -> str | None:
    """KIS quote API 의 prdt_abrv_name / hts_kor_isnm 으로 종목명 조회."""
    client = get_kis_client()
    if client is None:
        return None
    try:
        q = client.quote(sym)
        out = q.get("output", {}) if isinstance(q, dict) else {}
        for key in ("prdt_abrv_name", "hts_kor_isnm", "bstp_kor_isnm"):
            n = (out.get(key) or "").strip()
            if n:
                return n
        return None
    except Exception:
        return None


def _get_name(sym: str, kis_name: str = "") -> str:
    """이름 우선순위: KIS balance 응답 > 영구 cache > hardcoded SYMBOL_NAMES > KIS quote 1회 호출 → 캐시 채움 > 종목코드."""
    if kis_name and kis_name.strip():
        return kis_name.strip()
    cache = _name_cache_singleton()
    if sym in cache:
        return cache[sym]
    if sym in SYMBOL_NAMES:
        cache[sym] = SYMBOL_NAMES[sym]
        _save_name_cache(cache)
        return SYMBOL_NAMES[sym]
    # 모르는 종목 — KIS quote 1회 (이후 영구 캐시)
    n = _resolve_name_via_kis(sym)
    if n:
        cache[sym] = n
        _save_name_cache(cache)
        return n
    return sym   # 최종 fallback — 종목코드


@st.cache_resource
def get_kis_client():
    cfg = KISConfig.from_env()
    if not cfg.app_key:
        return None
    client = KISClient(cfg)
    try:
        client.token()
        return client
    except Exception:
        return None


def _i(s: str | None) -> int:
    """KIS 응답 string → int (콤마 제거, 빈 값 0)."""
    return int(str(s or "0").replace(",", "") or "0")


def _f(s: str | None) -> float:
    return float(str(s or "0").replace(",", "") or "0")


@st.cache_data(ttl=30)
def fetch_kis_balance() -> dict | None:
    """KIS 실제 계좌 잔고 — 진짜 truth source.

    Field 정의 (KIS docs 기반):
      total_eval: 총평가금액 (KIS 화면 "총평가금액")
      cash_total: 예수금 총액 (음수 = vts phantom margin debt)
      scts_eval:  유가증권 평가합 (보유 종목 평가)
      nxdy_settle: 익일정산금액 (T+1)
      d2_settle:   D+2 정산금액
      ord_avail:   T+2 정산 후 추정 가용 = cash_total + nxdy + d2
      pnl:         평가손익 합계
      pnl_rate:    수익률 (%)
      holdings:    [{code, name, qty, avg_price, cur_price, eval_amt, pnl_amt, pnl_pct}]
    """
    client = get_kis_client()
    if client is None:
        return None
    try:
        b = client.balance()
        if b.get("rt_cd") != "0":
            return {"_error": f"rt_cd={b.get('rt_cd')} {b.get('msg1','')}"}
        out2 = b.get("output2", [{}])[0] if b.get("output2") else {}
        out1 = b.get("output1") or []
        cash_total = _i(out2.get("dnca_tot_amt"))
        nxdy = _i(out2.get("nxdy_excc_amt"))
        d2 = _i(out2.get("prvs_rcdl_excc_amt"))
        total_eval = _i(out2.get("tot_evlu_amt"))
        bfdy_total = _i(out2.get("bfdy_tot_asst_evlu_amt"))
        # 당일 수익률 = (오늘 총평가 - 어제 마감 총평가) / 어제 마감
        daily_pnl = total_eval - bfdy_total if bfdy_total > 0 else 0
        daily_rate = (daily_pnl / bfdy_total * 100) if bfdy_total > 0 else 0.0
        return {
            "total_eval":   total_eval,
            "bfdy_total":   bfdy_total,             # 전일 총자산
            "daily_pnl":    daily_pnl,              # 당일 손익 (KRW)
            "daily_rate":   daily_rate,             # 당일 수익률 (%)
            "cash_total":   cash_total,
            "nxdy_settle":  nxdy,
            "d2_settle":    d2,
            "ord_avail":    cash_total + nxdy + d2, # T+2 정산 후 추정
            "scts_eval":    _i(out2.get("scts_evlu_amt")),
            "pnl":          _i(out2.get("asst_icdc_amt")),    # 누적 (참고)
            "pnl_rate":     _f(out2.get("asst_icdc_erng_rt")), # 누적 (참고)
            "holdings": [
                {
                    "code": h.get("pdno", ""),
                    "name": (h.get("prdt_name") or "").strip(),
                    "qty": _i(h.get("hldg_qty")),
                    "avg_price": _f(h.get("pchs_avg_pric")),
                    "cur_price": _f(h.get("prpr")),
                    "eval_amt": _i(h.get("evlu_amt")),
                    "pnl_amt": _i(h.get("evlu_pfls_amt")),
                    "pnl_pct": _f(h.get("evlu_pfls_rt")),
                }
                for h in out1
                if _i(h.get("hldg_qty")) != 0
            ],
        }
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:120]}"}


# 모듈 레벨 fallback — 직전 성공한 가집계 결과 보관 (KIS 일시 장애 시 재사용)
_LAST_SIGNALS = {
    "buy": pd.DataFrame(),
    "sell": pd.DataFrame(),
    "ratios": {},
    "ts": None,
    "stale": False,
}


@st.cache_data(ttl=120)
def fetch_dual_signals():
    """매수+매도 상위 30종 가집계 + dual 통과 분리.

    KIS API 가 일시적 connection abort 자주 (vts 특성) → 2회 retry + 실패 시
    직전 성공 결과 fallback. 즉 화면 표가 갑자기 모두 "—" 로 비는 일 방지.

    Returns: (buy_pass, sell_pass, all_ratios_dict)
    """
    client = get_kis_client()
    if client is None:
        return pd.DataFrame(), pd.DataFrame(), {}

    last_exc = None
    for attempt in range(3):
        try:
            buy_p = client.foreign_institution_total(market_code="0001", rank_sort="0")
            time.sleep(1.5)
            sell_p = client.foreign_institution_total(market_code="0001", rank_sort="1")
            df_buy = compute_flow_ratios(parse_foreign_institution_response(buy_p))
            df_sell = compute_flow_ratios(parse_foreign_institution_response(sell_p))
            if df_buy.empty and df_sell.empty:
                raise RuntimeError("both buy/sell empty (KIS 부분 응답?)")

            buy_pass = df_buy[
                (df_buy["flow_ratio"] > 0.05) & (df_buy["inst_ratio"] > 0.05)
            ].copy() if not df_buy.empty else df_buy.copy()
            sell_pass = df_sell[
                (df_sell["flow_ratio"] < -0.05) & (df_sell["inst_ratio"] < -0.05)
            ].copy() if not df_sell.empty else df_sell.copy()

            all_df = pd.concat([df_buy, df_sell]).drop_duplicates("symbol")
            all_ratios = {
                r["symbol"]: {
                    "flow_ratio": float(r["flow_ratio"]),
                    "inst_ratio": float(r["inst_ratio"]),
                    "name": r.get("name", ""),
                    "price": int(r.get("price", 0)),
                }
                for _, r in all_df.iterrows()
            }
            _LAST_SIGNALS["buy"] = buy_pass
            _LAST_SIGNALS["sell"] = sell_pass
            _LAST_SIGNALS["ratios"] = all_ratios
            _LAST_SIGNALS["ts"] = datetime.now()
            _LAST_SIGNALS["stale"] = False
            return buy_pass, sell_pass, all_ratios
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))   # 1.5, 3 s backoff

    # 3회 다 실패 → 직전 성공 결과 fallback
    _LAST_SIGNALS["stale"] = True
    if _LAST_SIGNALS["ts"] is not None:
        age = (datetime.now() - _LAST_SIGNALS["ts"]).total_seconds()
        st.info(
            f"📡 KIS 가집계 일시 장애 ({last_exc.__class__.__name__}) — "
            f"{int(age)}s 전 직전 결과 재사용 (표 비지 않게)"
        )
    else:
        st.warning(f"signal fetch failed (재시도 3회): {last_exc}")
    return _LAST_SIGNALS["buy"], _LAST_SIGNALS["sell"], _LAST_SIGNALS["ratios"]


def _sell_status(flow: float | None, inst: float | None) -> tuple[str, str]:
    """보유 종목 매도 임박도 평가.

    Returns: (label, color_emoji)
      "매도 신호"   🔴 — 둘 다 -5% 미만 (즉시 청산 트리거)
      "매도 압력"   🟠 — 둘 다 -2.5% ~ -5% (한 쪽 -5% 넘었거나 비슷)
      "약한 매도"   🟡 — 한 쪽만 부정적
      "정상"        🟢 — 매수/매도 양쪽 다 강하지 않음
      "데이터 없음" ⚪ — 60종 가집계 안에 없음
    """
    if flow is None or inst is None:
        return "데이터 없음", "⚪"
    if flow < -0.05 and inst < -0.05:
        return "매도 신호", "🔴"
    if flow < -0.025 and inst < -0.025:
        return "매도 압력", "🟠"
    if flow < 0 or inst < 0:
        return "약한 매도", "🟡"
    return "정상", "🟢"


def load_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


# --- 매매 이력 -----------------------------------------------------------
# runner 는 dual_trading.log 에 매매를 텍스트 한 줄로 찍는다.
# dual_trades.parquet (있으면) → 정확. 없으면 로그를 정규식으로 회수.
_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
_BUY_RE = re.compile(
    _TS + r".*?\[INFO\]\s+매수:\s+(\S+)\s+(.+?)\s+(\d+)주\s+@\s+([\d,]+)원"
    r"(?:\s*\(flow\s+([+-]?[\d.]+)%\s*/\s*inst\s+([+-]?[\d.]+)%[^)]*\))?"
)
_SELL_RE = re.compile(
    _TS + r".*?\[INFO\]\s+매도:\s+(\S+)\s+(\d+)주\s+@\s+([\d,]+)원"
)
_FORCE_RE = re.compile(
    _TS + r".*?\[INFO\]\s+강제 청산:\s+(\S+)\s+(\d+)주\s+@\s+([\d,]+)원"
)
_MDD_RE = re.compile(
    _TS + r".*?\[WARNING\] !!! MDD CAP 발동 \(([+-]?[\d.]+)% < -([\d.]+)%\)"
)


def _safe_read_tail(path: Path, max_bytes: int = 2_000_000) -> str:
    """봇이 들고 있어도 lock 안 걸리게 FileShare=ReadWrite 로 tail 읽기."""
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                _ = f.readline()   # 라인 경계 맞추기
            return f.read().decode("utf-8", errors="replace")
    except PermissionError:
        # Windows에서 봇이 쓰는 중일 때 — .NET FileStream 으로 ReadWrite 공유
        import ctypes
        try:
            from ctypes import wintypes
            GENERIC_READ = 0x80000000
            FILE_SHARE_RW = 0x00000003
            OPEN_EXISTING = 3
            h = ctypes.windll.kernel32.CreateFileW(
                str(path), GENERIC_READ, FILE_SHARE_RW, None, OPEN_EXISTING, 0, None
            )
            if h == -1:
                return ""
            # 그냥 os.fdopen 으로 못 가져와서 다시 try open with python io 양보
            ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            pass
        # python 으로 한 번 더 — share violation 일반화 회피
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-max_bytes:]
        except Exception:
            return ""


@st.cache_data(ttl=30)
def load_trades(_mtime: float) -> pd.DataFrame:
    """매매 이력 — parquet 우선, 없으면 trading.log 파싱.

    추가: 시간 오름차순으로 FIFO 큐 돌려서 매도 row 에 avg_entry / pnl_pct 채움.
    """
    # 1) parquet 우선 (runner 가 trades.parquet 에 row-level 저장 시작한 이후)
    rows: list[dict] = []
    if TRADES_PATH.exists():
        try:
            df = pd.read_parquet(TRADES_PATH)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
            # 누락 컬럼 None 으로 채워 backward compat 보장
            for col in ("flow", "inst", "avg_entry", "pnl_pct", "pnl_krw", "name"):
                if col not in df.columns:
                    df[col] = None
            rows = df.to_dict("records")
        except Exception:
            rows = []

    # 2) 로그 파싱 fallback
    if not rows:
        text = _safe_read_tail(TRADING_LOG, max_bytes=4_000_000)
        if not text:
            return pd.DataFrame()

        for m in _BUY_RE.finditer(text):
            ts, code, name, qty, price, flow, inst = m.groups()
            rows.append({
                "timestamp": ts, "side": "BUY", "code": code,
                "name": name.strip(),
                "qty": int(qty), "price": int(price.replace(",", "")),
                "flow": float(flow) if flow else None,
                "inst": float(inst) if inst else None,
            })
        for m in _SELL_RE.finditer(text):
            ts, code, qty, price = m.groups()
            rows.append({
                "timestamp": ts, "side": "SELL", "code": code,
                "name": SYMBOL_NAMES.get(code, code),
                "qty": int(qty), "price": int(price.replace(",", "")),
                "flow": None, "inst": None,
            })
        for m in _FORCE_RE.finditer(text):
            ts, code, qty, price = m.groups()
            rows.append({
                "timestamp": ts, "side": "FORCE_SELL", "code": code,
                "name": SYMBOL_NAMES.get(code, code),
                "qty": int(qty), "price": int(price.replace(",", "")),
                "flow": None, "inst": None,
            })

    if not rows:
        return pd.DataFrame()

    for r in rows:
        if not isinstance(r["timestamp"], pd.Timestamp):
            r["timestamp"] = pd.to_datetime(r["timestamp"])

    # 시간 오름차순 → FIFO 매수가 추적 → 매도 행에 avg_entry, pnl_pct, pnl_krw 채움
    rows.sort(key=lambda r: r["timestamp"])
    holdings: dict[str, dict] = {}   # code -> {qty, total_cost}
    for r in rows:
        code = r["code"]
        qty = int(r["qty"])
        price = float(r["price"])
        if r["side"] == "BUY":
            h = holdings.setdefault(code, {"qty": 0, "total_cost": 0.0})
            h["qty"] += qty
            h["total_cost"] += qty * price
            r["avg_entry"] = price   # 정보용 (매수 시점)
            r["pnl_pct"] = None
            r["pnl_krw"] = None
        else:   # SELL / FORCE_SELL
            h = holdings.get(code)
            if h and h["qty"] > 0:
                avg_entry = h["total_cost"] / h["qty"]
                sell_qty = min(qty, h["qty"])
                pnl_krw = sell_qty * (price - avg_entry)
                pnl_pct = (price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                r["avg_entry"] = avg_entry
                r["pnl_pct"] = pnl_pct
                r["pnl_krw"] = pnl_krw
                # FIFO pop 비례 차감
                h["total_cost"] -= sell_qty * avg_entry
                h["qty"] -= sell_qty
                if h["qty"] <= 0:
                    holdings.pop(code, None)
            else:
                r["avg_entry"] = None
                r["pnl_pct"] = None
                r["pnl_krw"] = None

    df = pd.DataFrame(rows)
    df["value"] = df["qty"] * df["price"]
    # name 보강 — 코드만 있거나 빈 경우 cache/KIS 로 채움
    df["name"] = df.apply(
        lambda r: _get_name(r["code"], "" if (
            pd.isna(r.get("name")) or not str(r.get("name", "")).strip()
            or str(r.get("name", "")).strip() == r["code"]
        ) else str(r["name"]).strip()),
        axis=1,
    )
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=30)
def load_mdd_events(_mtime: float) -> pd.DataFrame:
    """MDD CAP 발동 이벤트 — 로그에서 회수."""
    text = _safe_read_tail(TRADING_LOG, max_bytes=4_000_000)
    if not text:
        return pd.DataFrame()
    rows = []
    for m in _MDD_RE.finditer(text):
        ts, dd, cap = m.groups()
        rows.append({"timestamp": ts, "dd_pct": float(dd), "cap_pct": float(cap)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp", ascending=False)


CONFIG_PATH = ROOT / "config" / "symbols.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"stock": {"whitelist": [], "blacklist": [],
                          "max_concurrent": 7, "market_code": "0001"}}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"stock": {}}


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                            encoding="utf-8")


def render_symbol_config_sidebar():
    """사이드바에 종목 화이트/블랙리스트 편집 UI."""
    st.sidebar.markdown("### ⚙️ 종목 설정")
    cfg = load_config()
    stock = cfg.get("stock", {})

    market = st.sidebar.selectbox(
        "시장",
        options=["0001", "1001"],
        format_func=lambda x: {"0001": "KOSPI", "1001": "KOSDAQ"}.get(x, x),
        index=["0001", "1001"].index(stock.get("market_code", "0001")),
        key="cfg_market",
    )

    max_conc = st.sidebar.number_input(
        "최대 동시 보유 종목",
        min_value=1, max_value=20,
        value=int(stock.get("max_concurrent", 7)),
        key="cfg_max",
    )

    st.sidebar.markdown("**Whitelist** (이 종목만 매수, 빈 줄 = 모두 허용)")
    wl_default = "\n".join(stock.get("whitelist") or [])
    wl_text = st.sidebar.text_area(
        "Whitelist",
        value=wl_default,
        height=100,
        placeholder="005930\n000660\n005380",
        label_visibility="collapsed",
        key="cfg_wl",
    )

    st.sidebar.markdown("**Blacklist** (절대 매수 안 함)")
    bl_default = "\n".join(stock.get("blacklist") or [])
    bl_text = st.sidebar.text_area(
        "Blacklist",
        value=bl_default,
        height=80,
        placeholder="000720\n015760",
        label_visibility="collapsed",
        key="cfg_bl",
    )

    if st.sidebar.button("💾 설정 저장", use_container_width=True):
        whitelist = [s.strip() for s in wl_text.splitlines() if s.strip()]
        blacklist = [s.strip() for s in bl_text.splitlines() if s.strip()]
        cfg.setdefault("stock", {})
        cfg["stock"]["market_code"] = market
        cfg["stock"]["max_concurrent"] = int(max_conc)
        cfg["stock"]["whitelist"] = whitelist
        cfg["stock"]["blacklist"] = blacklist
        save_config(cfg)
        st.sidebar.success("✓ 저장됨. 다음 봇 iter (최대 2분 안) 자동 반영")

    st.sidebar.caption(
        f"현재: WL {len(stock.get('whitelist') or [])}종 / "
        f"BL {len(stock.get('blacklist') or [])}종 / "
        f"max {stock.get('max_concurrent', 7)}"
    )


def main():
    st.set_page_config(
        page_title="Dual Confirmation Paper Trading",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",   # 사이드바 기본 열림 (종목 설정용)
    )

    st.title("📊 Dual Confirmation Paper Trading")
    st.caption(f"외국인+기관 동방향 신호 동적 universe · max_concurrent=7 · MDD cap 25%")

    # ===== 봇 alive 상태 (heartbeat) =====
    hb_msg = None
    hb_age = None
    if HEARTBEAT_PATH.exists():
        try:
            hb_age = time.time() - HEARTBEAT_PATH.stat().st_mtime
            hb_msg = HEARTBEAT_PATH.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    hb_col, _ = st.columns([1, 4])
    with hb_col:
        if hb_age is None:
            st.error("🔴 봇 heartbeat 없음 — 봇이 한 번도 안 돌았거나 죽음")
        elif hb_age < 180:
            st.success(f"🟢 봇 alive · {int(hb_age)}s 전 heartbeat")
        elif hb_age < 600:
            st.warning(f"🟡 봇 stale · {int(hb_age)}s 전 heartbeat")
        else:
            st.error(f"🔴 봇 DEAD 의심 · {int(hb_age)}s 전 (>10분) — 확인 필요")
        if hb_msg:
            st.caption(hb_msg.replace("\n", " · "))

    # ===== KIS API circuit 상태 =====
    if CIRCUIT_PATH.exists():
        try:
            circ = json.loads(CIRCUIT_PATH.read_text(encoding="utf-8"))
            errs = circ.get("errors_in_window", 0)
            thr = circ.get("threshold", 10)
            win = circ.get("window_sec", 600)
            halted = circ.get("halted", False)
            remain = circ.get("halted_remaining_sec", 0)
            c1, c2 = st.columns([1, 4])
            with c1:
                if halted:
                    st.error(f"⛔ Circuit HALT · {remain}s 남음 — 매수 차단")
                elif errs == 0:
                    st.success(f"📡 KIS API 정상 · {win//60}분 윈도우 에러 0")
                elif errs < thr / 2:
                    st.info(f"📡 KIS API · {win//60}분 내 에러 {errs}/{thr}회")
                else:
                    st.warning(f"📡 KIS API 경고 · {win//60}분 내 에러 {errs}/{thr}회")
        except Exception:
            pass

    # 사이드바 — 종목 설정 UI
    render_symbol_config_sidebar()

    # auto-refresh
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()

    state = load_state()
    if state is None:
        st.error(f"state 파일 없음: {STATE_PATH}\n\n`python scripts/run_dual_paper_trading.py --dry-only` 먼저 실행")
        return

    history = load_history()

    # === KIS 실제 잔고 — 유일한 truth source ===
    kis_bal = fetch_kis_balance()
    if not kis_bal or kis_bal.get("_error"):
        err = kis_bal.get("_error") if kis_bal else "KIS client not initialized"
        st.error(f"⚠️ KIS 잔고 조회 실패: {err}\n\n5초 후 자동 재시도. KIS vts 일시 장애 시 자주 발생.")
        time.sleep(REFRESH_SEC)
        st.rerun()
        return

    initial = state.get("initial_cash", 10_000_000)
    peak = state.get("portfolio_peak", initial)   # state 는 peak 추적용으로만 사용

    kis_total = kis_bal["total_eval"]
    cash_total = kis_bal["cash_total"]
    ord_avail = kis_bal["ord_avail"]
    kis_n_pos = len(kis_bal["holdings"])
    daily_rate = kis_bal["daily_rate"]
    daily_pnl = kis_bal["daily_pnl"]
    bfdy_total = kis_bal["bfdy_total"]
    cum_rate = kis_bal["pnl_rate"]   # 누적 수익률 (참고)
    dd_pct = (kis_total - peak) / peak * 100 if peak > 0 else 0

    # === 상단 KPI ===
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Total", f"{kis_total:,}원",
                f"{daily_pnl:+,}원 ({daily_rate:+.2f}%)",
                help=f"전일 마감 {bfdy_total:,}원 대비 당일 변동\n(누적 수익률: {cum_rate:+.4f}%)")
    col2.metric(
        "가용현금 (예수금)",
        f"{cash_total:,}원",
        delta=f"T+2 정산 후 {ord_avail:,}원",
        delta_color="off",
        help=(f"지금 바로 매수에 쓸 수 있는 현금 = {cash_total:,}원 (예수금)\n"
              f"T+2 후 가용 = 예수금 {cash_total:,} + 익일정산 {kis_bal['nxdy_settle']:,} "
              f"+ D+2 {kis_bal['d2_settle']:,} = {ord_avail:,}원")
    )
    col3.metric("보유 종목", f"{kis_n_pos}/7")
    col4.metric("Drawdown", f"{dd_pct:+.2f}%", help=f"peak {peak:,.0f}원 (runner 추적)")

    # 봇 state.cash vs KIS 예수금 비교 (drift 보이게)
    bot_cash = float(state.get("cash", 0))
    drift = bot_cash - cash_total
    cap = f"봇 추정 현금: {bot_cash:,.0f}원"
    if abs(drift) > 1000:
        cap += f"  ·  ⚠ KIS 와 {drift:+,.0f}원 차이 (다음 iter sync 가 정정)"
    else:
        cap += f"  ·  ✓ KIS 와 일치"
    st.caption(cap)

    # vts phantom debt 경고
    if cash_total < 0:
        st.error(
            f"⚠️ **PHANTOM MARGIN DEBT** — 예수금 {cash_total:,}원 (음수). "
            f"vts 가 가용 현금 안 보고 매수 받아준 결과. "
            f"T+2 정산 ({kis_bal['d2_settle']:+,}원) 후 자동 해소 예상."
        )

    cooldown = state.get("cooldown_remaining", 0)
    if cooldown > 0:
        st.warning(f"⚠️ MDD Cap Cooldown: {cooldown}일 남음 — 매수 차단")

    last_run = state.get("last_run", "—")
    run_count = state.get("run_count", 0)
    st.caption(f"마지막 실행 (runner): {last_run} | 누적 실행 횟수: {run_count}")

    st.divider()

    # === 실시간 신호 (보유 종목 매도 임박도 분석용으로 미리 호출) ===
    buy_pass, sell_pass, all_ratios = fetch_dual_signals()

    # === 보유 종목 (KIS truth) ===
    st.subheader("📦 보유 종목 + 매도 임박도")
    holdings_list = kis_bal["holdings"]
    state_positions = state.get("positions", {})   # 진입일 lookup 용

    if not holdings_list:
        st.info("보유 종목 없음")
    else:
        rows = []
        for h in holdings_list:
            sym = h["code"]
            name = h["name"] or _get_name(sym)

            # 진입일 — state 에서 lookup (없으면 —)
            entry_date = state_positions.get(sym, {}).get("entry_date", "—")

            # 매도 임박도 (실시간 가집계)
            r_info = all_ratios.get(sym)
            if r_info:
                flow = r_info["flow_ratio"]
                inst = r_info["inst_ratio"]
                label, emoji = _sell_status(flow, inst)
                sell_str = f"{emoji} {label}"
                ratios_str = f"외인 {flow*100:+.1f}% / 기관 {inst*100:+.1f}%"
            else:
                # 가집계 상위 30 밖 → investor_trend fallback (전영업일 순매수 주수)
                fb = _investor_trend_for(sym)
                if fb:
                    flow = fb["flow_ratio"]
                    inst = fb["inst_ratio"]
                    label, emoji = _sell_status(flow, inst)
                    sell_str = f"{emoji} {label} (전영업일)"
                    ratios_str = f"외인 {flow*100:+.1f}% / 기관 {inst*100:+.1f}%"
                else:
                    sell_str = "🟢 정상"
                    ratios_str = "(상위 30 외)"

            rows.append({
                "종목": name,
                "코드": sym,
                "수량": h["qty"],
                "매입가": f"{h['avg_price']:,.0f}",
                "현재가": f"{h['cur_price']:,.0f}",
                "평가금액": f"{h['eval_amt']:,}",
                "손익": f"{h['pnl_amt']:+,}",
                "손익률": f"{h['pnl_pct']:+.2f}%",
                "매도신호": sell_str,
                "외인/기관": ratios_str,
                "진입일": entry_date,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("ℹ️ KIS 계좌 실제 holdings · 가격/평가/손익 모두 KIS 실시간 데이터")

    st.divider()

    # === 오늘 신호 (위에서 이미 호출, 캐시 hit) ===
    st.subheader("🔔 실시간 Dual Confirmation 신호")
    col_buy, col_sell = st.columns(2)

    with col_buy:
        st.markdown(f"**매수 후보** (외인+기관 둘 다 +5%↑) — {len(buy_pass)}종")
        if buy_pass.empty:
            st.caption("없음")
        else:
            df_show = buy_pass[["name", "symbol", "price", "change_pct", "flow_ratio", "inst_ratio"]].copy()
            df_show["name"] = df_show.apply(lambda r: r["name"] if r["name"] else _get_name(r["symbol"]), axis=1)
            df_show["change_pct"] = df_show["change_pct"].apply(lambda x: f"{x:+.2f}%")
            df_show["flow_ratio"] = df_show["flow_ratio"].apply(lambda x: f"{x*100:+.2f}%")
            df_show["inst_ratio"] = df_show["inst_ratio"].apply(lambda x: f"{x*100:+.2f}%")
            df_show.columns = ["종목", "코드", "현재가", "당일등락", "외인%", "기관%"]
            st.dataframe(df_show, use_container_width=True, hide_index=True)

    with col_sell:
        st.markdown(f"**매도 후보** (외인+기관 둘 다 -5%↓) — {len(sell_pass)}종")
        if sell_pass.empty:
            st.caption("없음")
        else:
            df_show = sell_pass[["symbol", "name", "price", "flow_ratio", "inst_ratio"]].copy()
            df_show["flow_ratio"] = df_show["flow_ratio"].apply(lambda x: f"{x*100:+.2f}%")
            df_show["inst_ratio"] = df_show["inst_ratio"].apply(lambda x: f"{x*100:+.2f}%")
            df_show.columns = ["종목", "이름", "현재가", "외인%", "기관%"]
            st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.divider()

    # === Portfolio 시계열 차트 ===
    st.subheader("📈 Portfolio 시계열")
    if history.empty:
        st.info("아직 충분한 history 없음 — 며칠 운용 후 차트 표시")
    else:
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.sort_values("timestamp")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history["timestamp"], y=history["portfolio_total"],
            name="Portfolio Total", line=dict(color="#0969da", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=history["timestamp"], y=history["portfolio_peak"],
            name="Peak (cummax)", line=dict(color="#bf8700", dash="dash", width=1),
        ))
        fig.add_hline(y=initial, line_dash="dot", line_color="gray",
                      annotation_text="시작 자본")
        fig.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # drawdown 차트
        st.markdown("**Drawdown**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=history["timestamp"], y=history["drawdown"] * 100,
            fill="tozeroy", line=dict(color="#cf222e"),
        ))
        fig2.add_hline(y=-30, line_dash="dot", line_color="red",
                       annotation_text="MDD Cap 30%")
        fig2.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10),
                           yaxis_title="DD %")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # === 매매 이력 ===
    st.subheader("📜 매매 이력")
    log_mtime = TRADING_LOG.stat().st_mtime if TRADING_LOG.exists() else 0.0
    trades = load_trades(log_mtime)
    mdd_events = load_mdd_events(log_mtime)

    src = "dual_trades.parquet" if TRADES_PATH.exists() else "dual_trading.log (파싱)"
    st.caption(f"source: {src} · 총 {len(trades)}건 · MDD 발동 {len(mdd_events)}회")

    if trades.empty:
        st.info("매매 이력 없음")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        n_buy = (trades["side"] == "BUY").sum()
        n_sell = (trades["side"] == "SELL").sum()
        n_force = (trades["side"] == "FORCE_SELL").sum()
        tot_value = int(trades["value"].sum())
        col_a.metric("총 매수", f"{n_buy}건")
        col_a.caption(f"{int(trades.loc[trades['side']=='BUY', 'value'].sum()):,}원")
        col_b.metric("총 매도", f"{n_sell}건")
        col_b.caption(f"{int(trades.loc[trades['side']=='SELL', 'value'].sum()):,}원")
        col_c.metric("강제 청산", f"{n_force}건",
                     delta=f"MDD {len(mdd_events)}회" if len(mdd_events) else None,
                     delta_color="inverse")
        col_d.metric("총 거래대금", f"{tot_value/1e8:.2f}억")

        # ===== 매도 PnL 통계 (FIFO 기반 추정) =====
        sell_mask = trades["side"].isin(["SELL", "FORCE_SELL"])
        sell_with_pnl = trades[sell_mask & trades["pnl_krw"].notna()].copy()
        if not sell_with_pnl.empty:
            pnl_total = float(sell_with_pnl["pnl_krw"].sum())
            pnl_mean_pct = float(sell_with_pnl["pnl_pct"].mean()) * 100
            n_win = int((sell_with_pnl["pnl_krw"] > 0).sum())
            n_lose = int((sell_with_pnl["pnl_krw"] < 0).sum())
            hit = n_win / (n_win + n_lose) * 100 if (n_win + n_lose) > 0 else 0
            best = float(sell_with_pnl["pnl_pct"].max()) * 100
            worst = float(sell_with_pnl["pnl_pct"].min()) * 100

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("매도 실현 PnL 합", f"{pnl_total:+,.0f}원",
                      delta=f"{pnl_total/1e4:+.1f}만원", delta_color="off")
            p2.metric("평균 매도 수익률", f"{pnl_mean_pct:+.2f}%")
            p3.metric("승률", f"{hit:.1f}%",
                      delta=f"{n_win}승 {n_lose}패", delta_color="off")
            p4.metric("최대 익절 / 최대 손절",
                      f"{best:+.2f}% / {worst:+.2f}%")

        # 사이드/일자 필터
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            sides = st.multiselect(
                "side", options=["BUY", "SELL", "FORCE_SELL"],
                default=["BUY", "SELL", "FORCE_SELL"],
            )
        with f_col2:
            n_show = st.number_input("표시 건수", min_value=10, max_value=2000,
                                     value=100, step=20)
        with f_col3:
            symbol_filter = st.text_input(
                "종목 (코드/이름 substring)", value="",
                placeholder="예: 005930 또는 삼성",
            )

        view = trades[trades["side"].isin(sides)].copy()
        if symbol_filter:
            sf = symbol_filter.strip()
            view = view[
                view["code"].str.contains(sf, na=False)
                | view["name"].astype(str).str.contains(sf, na=False)
            ]
        view = view.head(int(n_show))

        # 컬러 매핑 (BUY=빨강 KR 관행 / SELL=파랑)
        # 한글 컬럼명으로 rename 한 뒤 styling 되니까 "구분" key 사용.
        def _style_row(r):
            side = r.get("구분", "")
            if side == "BUY":
                bg = "background-color: #fff0f0"
            elif side == "SELL":
                bg = "background-color: #f0f4ff"
            else:
                bg = "background-color: #fff7d6"
            return [bg] * len(r)

        show = view.copy()
        show["timestamp"] = show["timestamp"].dt.strftime("%m-%d %H:%M:%S")
        show["price"] = show["price"].apply(lambda v: f"{v:,}")
        show["value"] = show["value"].apply(lambda v: f"{v:,}")
        show["flow"] = show["flow"].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "")
        show["inst"] = show["inst"].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "")
        show["avg_entry"] = show["avg_entry"].apply(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "")
        show["pnl_pct"] = show["pnl_pct"].apply(
            lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "")
        show["pnl_krw"] = show["pnl_krw"].apply(
            lambda v: f"{v:+,.0f}" if pd.notna(v) else "")
        show = show[["timestamp", "side", "code", "name", "qty",
                     "price", "avg_entry", "pnl_pct", "pnl_krw",
                     "value", "flow", "inst"]]
        show.columns = ["시각", "구분", "코드", "종목", "수량",
                        "가격", "매수가", "수익률", "수익(원)",
                        "금액", "외인%", "기관%"]
        st.dataframe(show.style.apply(_style_row, axis=1),
                     use_container_width=True, hide_index=True, height=420)

        # 일별 매매 횟수 막대
        st.markdown("**일별 매매 빈도**")
        daily = trades.copy()
        daily["date"] = daily["timestamp"].dt.date
        agg = daily.groupby(["date", "side"]).size().unstack(fill_value=0)
        for col in ("BUY", "SELL", "FORCE_SELL"):
            if col not in agg.columns:
                agg[col] = 0
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=agg.index, y=agg["BUY"], name="BUY",
                               marker_color="#cf222e"))
        fig_d.add_trace(go.Bar(x=agg.index, y=agg["SELL"], name="SELL",
                               marker_color="#0969da"))
        fig_d.add_trace(go.Bar(x=agg.index, y=agg["FORCE_SELL"], name="FORCE_SELL",
                               marker_color="#bf8700"))
        # MDD 발동 시점 표시.
        # plotly add_vline + annotation 조합이 datetime axis 에서 sum(Timestamp) 를
        # 내부 호출해 깨짐 (pandas 2.x). 안전한 방법: add_shape (line) 만 그리고
        # annotation 은 별도로 add_annotation. x 는 date string 으로 통일.
        if not mdd_events.empty:
            for d in mdd_events["timestamp"].dt.strftime("%Y-%m-%d").unique():
                fig_d.add_shape(
                    type="line", x0=d, x1=d, y0=0, y1=1,
                    yref="paper", xref="x",
                    line=dict(color="red", dash="dot", width=1),
                )
                fig_d.add_annotation(
                    x=d, y=1, yref="paper", xref="x",
                    text="MDD", showarrow=False,
                    font=dict(color="red", size=10),
                    yanchor="bottom",
                )
        fig_d.update_layout(barmode="stack", height=240,
                            margin=dict(t=10, b=10, l=10, r=10),
                            xaxis_title="date", yaxis_title="건수")
        st.plotly_chart(fig_d, use_container_width=True)

    st.divider()
    st.caption(f"자동 새로고침 {REFRESH_SEC}초 | KIS env={os.environ.get('KIS_ENV', 'vts')}")

    # auto refresh
    time.sleep(REFRESH_SEC)
    st.rerun()


if __name__ == "__main__":
    main()
