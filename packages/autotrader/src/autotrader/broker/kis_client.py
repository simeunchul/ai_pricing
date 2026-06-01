"""KIS Developers REST client.

Docs: https://apiportal.koreainvestment.com/

IMPORTANT: every order-submitting method checks `config.dry_run`. When True,
the method returns a mocked response without hitting the order endpoint.
Live trading requires explicitly setting KIS_DRY_RUN=false AND KIS_ENV=prod.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


VTS_HOST = "https://openapivts.koreainvestment.com:29443"   # 모의투자
PROD_HOST = "https://openapi.koreainvestment.com:9443"      # 실계좌


@dataclass
class KISConfig:
    app_key: str
    app_secret: str
    account_no: str                       # 8-digit
    account_prod: str = "01"              # 01 = cash
    env: str = "vts"                      # "vts" | "prod"
    dry_run: bool = True
    token_cache: str = ".kis_token.json"

    @classmethod
    def from_env(cls) -> "KISConfig":
        env = os.environ.get("KIS_ENV", "vts")
        dry = os.environ.get("KIS_DRY_RUN", "true").lower() != "false"
        # ONLY allow live when explicitly prod + dry_run=false
        if env == "prod" and dry is False:
            logger.warning("[KIS] LIVE trading config enabled.")
        return cls(
            app_key=os.environ.get("KIS_APP_KEY", ""),
            app_secret=os.environ.get("KIS_APP_SECRET", ""),
            account_no=os.environ.get("KIS_ACCOUNT", ""),
            account_prod=os.environ.get("KIS_ACCOUNT_PROD", "01"),
            env=env,
            dry_run=dry,
        )

    @property
    def host(self) -> str:
        return PROD_HOST if self.env == "prod" else VTS_HOST


class KISClient:
    def __init__(self, cfg: KISConfig):
        self.cfg = cfg
        self._token: str | None = None
        self._token_exp: float = 0.0
        self.session = requests.Session()

    # ---------------------------------------------------------------
    # OAuth
    # ---------------------------------------------------------------
    def _load_cached_token(self) -> None:
        p = Path(self.cfg.token_cache)
        if not p.exists():
            return
        try:
            d = json.loads(p.read_text())
            if d.get("host") != self.cfg.host:
                return
            if d.get("expires_at", 0) > time.time() + 300:
                self._token = d["access_token"]
                self._token_exp = d["expires_at"]
        except Exception:
            pass

    def _save_token(self) -> None:
        if self._token:
            Path(self.cfg.token_cache).write_text(json.dumps({
                "host": self.cfg.host,
                "access_token": self._token,
                "expires_at": self._token_exp,
            }))

    def token(self) -> str:
        if not self._token or time.time() > self._token_exp:
            self._load_cached_token()
        if self._token and time.time() < self._token_exp:
            return self._token

        url = f"{self.cfg.host}/oauth2/tokenP"
        r = self.session.post(
            url,
            json={
                "grant_type": "client_credentials",
                "appkey": self.cfg.app_key,
                "appsecret": self.cfg.app_secret,
            },
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        self._token = d["access_token"]
        self._token_exp = time.time() + int(d.get("expires_in", 86400)) - 600
        self._save_token()
        return self._token

    def _headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.token()}",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    # ---------------------------------------------------------------
    # Queries
    # ---------------------------------------------------------------
    def quote(self, symbol: str) -> dict:
        """Current price inquiry (주식현재가)."""
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/quotations/inquire-price"
        tr = "FHKST01010100"
        r = self.session.get(
            url,
            headers=self._headers(tr),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def index_price(self, index_code: str = "0001") -> dict:
        """국내 업종/지수 현재가 (FHPUP02100000).

        Args:
            index_code: "0001" KOSPI 종합, "1001" KOSDAQ 종합, "2001" KOSPI200.

        Response output 주요 필드:
          bstp_nmix_prpr        업종 지수 현재가
          bstp_nmix_prdy_vrss   전일 대비
          bstp_nmix_prdy_ctrt   전일 대비 등락률 (%)
          bstp_nmix_oprc        시가
          bstp_nmix_hgpr        고가
          bstp_nmix_lwpr        저가

        Underperform 룰 (#5) 의 코스피 ref price 원천. 매 iter 1회 호출.
        """
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/quotations/inquire-index-price"
        tr = "FHPUP02100000"
        r = self.session.get(
            url,
            headers=self._headers(tr),
            params={"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": index_code},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def investor_trend(self, symbol: str) -> dict:
        """종목별 외국인/기관 매매 동향 (FHKST01010900).

        TR: FHPST01010900 (vts) / FHKST01010900 (prod). 응답 output 은 일자별
        외국인/기관 순매수 (-N일 ~ -1일) 리스트. 종목별 외인/기관 누적 시계열
        획득용. 가집계 상위 30종 밖 보유 종목의 외인/기관 비율 조회 시 사용.
        """
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/quotations/inquire-investor"
        tr = "FHKST01010900"
        r = self.session.get(
            url,
            headers=self._headers(tr),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def minute_chart(
        self,
        symbol: str,
        time_hhmmss: str = "153000",
        include_past: str = "Y",
    ) -> dict:
        """Intraday minute chart from anchor time (FHKST03010200).

        Returns ~30 most recent 1-minute bars before time_hhmmss on TODAY.
        For multi-day historical backfill, walk anchor time backwards or use
        minute_chart_day() with explicit date.

        Response output2 fields per bar (KIS spec):
          stck_bsop_date, stck_cntg_hour, stck_oprc, stck_hgpr, stck_lwpr,
          stck_prpr, cntg_vol, acml_tr_pbmn
        """
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        tr = "FHKST03010200"
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_HOUR_1": time_hhmmss,
            "FID_PW_DATA_INCU_YN": include_past,
        }
        r = self.session.get(url, headers=self._headers(tr), params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def minute_chart_day(self, symbol: str, date_yyyymmdd: str) -> dict:
        """Full-day minute chart for a historical date (FHKST03010230).

        Some KIS environments name this differently or restrict to prod —
        if it 404s on vts, caller should walk minute_chart() backwards.
        """
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
        tr = "FHKST03010230"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": date_yyyymmdd,
            "FID_INPUT_HOUR_1": "153000",
            "FID_PW_DATA_INCU_YN": "Y",
            "FID_FAKE_TICK_INCU_YN": "N",
        }
        r = self.session.get(url, headers=self._headers(tr), params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def foreign_institution_total(
        self,
        market_div: str = "V",          # 시장구분 (KIS 화면 V 가 외국인기관 전용)
        market_code: str = "0001",      # "0001" KOSPI, "1001" KOSDAQ — "0000" 은 500 에러
        rank_sort: str = "0",           # "0" 순매수상위 / "1" 순매도상위
        div_cls: str = "0",             # "0" 수량정렬 / "1" 금액정렬
    ) -> dict:
        """시장별 외국인/기관 매매종목 가집계 — 영웅문 장중 외국인 화면 데이터.

        KIS Endpoint: /uapi/domestic-stock/v1/quotations/foreign-institution-total
        TR_ID:        FHPTJ04400000

        장중 약 1분 지연 추정치. 정확도는 EOD inquire-investor 보다 떨어지지만
        실시간성 (T+1 시가 기다리지 않고 즉시 반응) 이 결정적 가치.

        Returns response with output[] list. 각 항목 주요 필드:
          - mksc_shrn_iscd        : 종목코드
          - hts_kor_isnm          : 종목명
          - stck_prpr             : 현재가
          - frgn_ntby_qty         : 외국인 순매수 수량
          - frgn_ntby_tr_pbmn     : 외국인 순매수 거래대금
          - orgn_ntby_qty         : 기관 순매수 수량
          - orgn_ntby_tr_pbmn     : 기관 순매수 거래대금
          - sum_ntby_qty          : 외국인+기관 합산 순매수 수량
          - sum_ntby_tr_pbmn      : 외국인+기관 합산 거래대금
        """
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/quotations/foreign-institution-total"
        tr = "FHPTJ04400000"
        params = {
            "FID_COND_MRKT_DIV_CODE": market_div,
            "FID_COND_SCR_DIV_CODE": "16449",
            "FID_INPUT_ISCD": market_code,
            "FID_DIV_CLS_CODE": div_cls,
            "FID_RANK_SORT_CLS_CODE": rank_sort,
            "FID_ETC_CLS_CODE": "0",
        }
        r = self.session.get(url, headers=self._headers(tr), params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def asking_price(self, symbol: str) -> dict:
        """5호가 + 예상 체결가 조회. 호가 스프레드 / 잔량 분석용.

        KIS Endpoint: /uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn
        TR_ID: FHKST01010200

        Output1 fields:
          askp1~askp10  : 매도 1~10호가 가격
          bidp1~bidp10  : 매수 1~10호가 가격
          askp_rsqn1~10 : 매도 잔량
          bidp_rsqn1~10 : 매수 잔량
          total_askp_rsqn / total_bidp_rsqn: 매도/매수 총 잔량
        """
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        tr = "FHKST01010200"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
        }
        r = self.session.get(url, headers=self._headers(tr), params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def balance(self) -> dict:
        """Cash balance inquiry (잔고)."""
        url = f"{self.cfg.host}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr = "VTTC8434R" if self.cfg.env == "vts" else "TTTC8434R"
        params = {
            "CANO": self.cfg.account_no,
            "ACNT_PRDT_CD": self.cfg.account_prod,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        r = self.session.get(url, headers=self._headers(tr), params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    # ---------------------------------------------------------------
    # Orders (GUARDED)
    # ---------------------------------------------------------------
    def order(
        self,
        symbol: str,
        qty: int,
        side: str,              # "buy" | "sell"
        price: int = 0,         # 0 = market
        ord_div: str = "00",    # "00"=지정가, "01"=시장가
    ) -> dict:
        """Submit cash order. Returns {mocked: True, ...} when dry_run."""
        if self.cfg.dry_run:
            mock = {
                "mocked": True,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "price": price,
                "ord_div": ord_div,
                "ts": time.time(),
            }
            logger.info(f"[KIS DRY_RUN] order: {mock}")
            return mock

        url = f"{self.cfg.host}/uapi/domestic-stock/v1/trading/order-cash"
        # tr_id differs for vts/prod and buy/sell
        if self.cfg.env == "vts":
            tr = "VTTC0802U" if side == "buy" else "VTTC0801U"
        else:
            tr = "TTTC0802U" if side == "buy" else "TTTC0801U"

        body = {
            "CANO": self.cfg.account_no,
            "ACNT_PRDT_CD": self.cfg.account_prod,
            "PDNO": symbol,
            "ORD_DVSN": ord_div,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price),
        }
        r = self.session.post(url, headers=self._headers(tr), json=body, timeout=10)
        r.raise_for_status()
        return r.json()


__all__ = ["KISConfig", "KISClient"]
