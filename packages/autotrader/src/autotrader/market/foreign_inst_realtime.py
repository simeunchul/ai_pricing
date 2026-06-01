"""KIS 실시간 외국인+기관 가집계 데이터 wrapper.

영웅문 HTS 의 "장중 외국인/기관 매매" 화면이 사용하는 KIS endpoint
(`foreign-institution-total`) 를 wrap 해서 strategy 가 일별 EOD 데이터 대신
**장중 1분 단위 추정치**로 동작하게 한다.

Workflow:
  1. KIS client 인증 (기존 token 재사용)
  2. 분 단위 polling (예: 60초)
  3. 응답 list → DataFrame 변환
  4. 종목별 frgn_ntby_qty / orgn_ntby_qty / volume 으로 flow_ratio / inst_ratio 계산
  5. Strategy.decide() 에 전달 (signal_keys=("flow_ratio", "inst_ratio"))

EOD inquire-investor (정밀) vs 장중 가집계 (추정):
  - 정확도: EOD 가 우월. 장중은 회원사 가집계라 ±5~10% 오차 가능
  - 실시간성: 장중이 압도. T+1 시가 대기 불필요
  - 활용: 백테스트 → EOD, 라이브 → 장중 추정치 + EOD 사후 보정
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from autotrader.broker.kis_client import KISClient

logger = logging.getLogger(__name__)


def parse_foreign_institution_response(payload: dict) -> pd.DataFrame:
    """KIS foreign-institution-total 응답 → DataFrame.

    실제 응답 형식 (장중 가집계, 시장별 순매수/순매도 상위 30 종목):
      hts_kor_isnm        종목명
      mksc_shrn_iscd      종목코드
      stck_prpr           현재가
      acml_vol            당일 누적 거래량  ⭐ (별도 quote 불필요)
      ntby_qty            외국인+기관 합계 순매수 수량
      frgn_ntby_qty       외국인 순매수 수량
      orgn_ntby_qty       기관 순매수 수량
      frgn_ntby_tr_pbmn   외국인 순매수 거래대금 (백만원 단위 추정)
      orgn_ntby_tr_pbmn   기관 순매수 거래대금
      ivtr_ntby_qty       투신 순매수
      bank_ntby_qty       은행 순매수
      insu_ntby_qty       보험 순매수
      fund_ntby_qty       펀드 순매수
      prdy_ctrt           등락률 %
    """
    output = payload.get("output", [])
    if not isinstance(output, list):
        return pd.DataFrame()

    rows = []
    for item in output:
        try:
            rows.append({
                "symbol": item.get("mksc_shrn_iscd", ""),
                "name": item.get("hts_kor_isnm", ""),
                "price": _safe_int(item.get("stck_prpr")),
                "volume": _safe_int(item.get("acml_vol")),
                "foreign_net_qty": _safe_int(item.get("frgn_ntby_qty")),
                "foreign_net_value": _safe_int(item.get("frgn_ntby_tr_pbmn")),
                "inst_net_qty": _safe_int(item.get("orgn_ntby_qty")),
                "inst_net_value": _safe_int(item.get("orgn_ntby_tr_pbmn")),
                "sum_net_qty": _safe_int(item.get("ntby_qty")),
                "change_pct": _safe_float(item.get("prdy_ctrt")),
            })
        except Exception as e:
            logger.debug(f"parse error: {e}")
            continue
    return pd.DataFrame(rows)


def _safe_float(s) -> float:
    if s is None:
        return 0.0
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace(",", "").replace("+", "")
    if not s or s == "-":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _safe_int(s) -> int:
    if s is None:
        return 0
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().replace(",", "")
    if not s or s == "-":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


@dataclass
class RealtimeFlowCache:
    """장중 외국인+기관 가집계 in-memory cache.

    Strategy 운용 중 1분 단위로 업데이트. 종목별 marginal flow 추적 가능.
    """
    last_update: float = 0.0
    last_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    update_interval: float = 60.0   # 초

    def get(
        self,
        client: KISClient,
        market_code: str = "0000",
        force: bool = False,
    ) -> pd.DataFrame:
        """캐시 만료 또는 force=True 시 KIS API 호출, 그 외 캐시 반환."""
        now = time.time()
        if not force and (now - self.last_update) < self.update_interval and not self.last_data.empty:
            return self.last_data.copy()

        try:
            payload = client.foreign_institution_total(market_code=market_code)
            df = parse_foreign_institution_response(payload)
            if not df.empty:
                self.last_data = df
                self.last_update = now
            return df.copy()
        except Exception as e:
            logger.warning(f"foreign_institution_total fetch failed: {e}")
            return self.last_data.copy() if not self.last_data.empty else pd.DataFrame()


def compute_flow_ratios(
    df: pd.DataFrame,
    daily_volumes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """flow_ratio / inst_ratio 계산.

    Args:
        df: parse_foreign_institution_response 결과 (volume 컬럼 포함)
        daily_volumes: optional override — 외부에서 거래량 주입.
                       None 이면 df['volume'] (응답의 acml_vol) 사용.

    Returns: df 에 'flow_ratio', 'inst_ratio' 컬럼 추가
    """
    # 빈 입력에도 downstream 이 'symbol'/'flow_ratio' 접근해도 KeyError 안 나도록
    # 명세 컬럼만 가진 empty DataFrame 반환.
    expected_cols = ["symbol", "name", "price", "volume",
                     "foreign_net_qty", "inst_net_qty", "change_pct",
                     "flow_ratio", "inst_ratio"]
    if df is None or df.empty:
        return pd.DataFrame(columns=expected_cols)

    out = df.copy()
    flow_ratios = []
    inst_ratios = []
    for _, row in out.iterrows():
        sym = row["symbol"]
        vol = (daily_volumes.get(sym, 0)
               if daily_volumes is not None
               else int(row.get("volume", 0)))
        if vol > 0:
            flow_ratios.append(row["foreign_net_qty"] / vol)
            inst_ratios.append(row["inst_net_qty"] / vol)
        else:
            flow_ratios.append(0.0)
            inst_ratios.append(0.0)
    out["flow_ratio"] = flow_ratios
    out["inst_ratio"] = inst_ratios
    return out


@dataclass
class PersistentPayloadCache:
    """Disk-backed last-good cache for foreign-institution-total payloads.

    Motivation: KIS 모의투자 서버 (`openapivts`) 의 `foreign-institution-total`
    엔드포인트는 매일 09:00~09:15 사이 500 에러 또는 빈 응답을 반환하는 패턴이
    관찰됨 (2026-05-19 ~ 05-26 로그). 시초 22분이 외국인 갭매수 알파의 절반
    이상을 차지하는 구간이라 이 공백을 메우는 것이 결정적.

    Strategy:
      - 가집계 호출이 성공 (어느 한쪽이라도 비어있지 않음) 하면 buy/sell payload
        쌍을 timestamp 와 함께 JSON 으로 disk 에 저장.
      - 시초 빈 응답 시 disk 에서 직전 거래일 payload 로드 → fallback 사용.
      - max_age_hours (기본 48h) 초과 stale 은 무시 (주말 4일짜리 stale 차단).
      - 시장구분(market_code) 별로 별도 파일 — KOSPI/KOSDAQ 혼동 방지.

    Tradeoff: stale 데이터는 진짜 오늘의 외국인 flow 가 아님. 따라서 caller 는
    fallback 사용 시 임계값을 강화 (예: ENTER_THRESHOLD * 1.5) 해서 false signal
    위험을 제한해야 한다. 본 클래스는 단순 저장/조회만 책임.
    """
    cache_dir: Path
    max_age_hours: float = 48.0

    def _path(self, market_code: str) -> Path:
        return self.cache_dir / f"dual_last_good_payload_{market_code}.json"

    def save(self, market_code: str, payload_buy: dict | None,
             payload_sell: dict | None) -> None:
        """가집계 응답을 disk 에 저장. 둘 다 None 이면 저장하지 않음."""
        if not payload_buy and not payload_sell:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "saved_at": datetime.now().isoformat(),
                "market_code": market_code,
                "payload_buy": payload_buy or {},
                "payload_sell": payload_sell or {},
            }
            tmp = self._path(market_code).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self._path(market_code))
        except Exception as e:
            logger.warning(f"PersistentPayloadCache save failed: {e}")

    def load(self, market_code: str) -> tuple[dict, dict, float] | None:
        """직전 good payload 로드.

        Returns: (payload_buy, payload_sell, age_hours) — fresh enough 한 경우.
                 stale 또는 파일 없음 → None.
        """
        p = self._path(market_code)
        if not p.exists():
            return None
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"PersistentPayloadCache load failed: {e}")
            return None

        try:
            saved_at = datetime.fromisoformat(record["saved_at"])
        except (KeyError, ValueError):
            return None
        age = datetime.now() - saved_at
        if age > timedelta(hours=self.max_age_hours):
            logger.info(
                f"PersistentPayloadCache stale ({age.total_seconds() / 3600:.1f}h "
                f"> {self.max_age_hours}h) — fallback 무시"
            )
            return None
        return (
            record.get("payload_buy") or {},
            record.get("payload_sell") or {},
            age.total_seconds() / 3600.0,
        )


__all__ = [
    "parse_foreign_institution_response",
    "compute_flow_ratios",
    "RealtimeFlowCache",
    "PersistentPayloadCache",
]
