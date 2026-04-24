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
