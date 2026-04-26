"""B2 → ELS daily NAV integration.

매일 시장 IV surface 받아 Heston calibrate (B2) → 추출한 vol 로 ELS 재가격 (Layer A).
한화 8286호 같은 multi-asset ELS 의 일일 fair value 자동 산출.

Pipeline:
  1. yfinance / KRX 에서 오늘 옵션 IV surface 받기 (per asset)
  2. B2 DeepCalibNet 으로 각 자산 calibrated Heston params
  3. v0 (instantaneous variance) 에서 spot vol σ_today = sqrt(v0)
  4. 이 σ 들을 Layer A price_els() 에 투입 → 오늘 fair value
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pricing.heston import HestonParams
from pricing.els.step_down import StepDownELS, price_els, ELSResult
from ai_pricing.deep_calib.model import DeepCalibNet, DeepCalibConfig
from ai_pricing.deep_calib.calibrate import calibrate as nn_calibrate


@dataclass
class ELSDailyNAVResult:
    fair_value_krw: float
    fair_value_stderr: float
    issue_price_krw: float
    deviation_pct: float
    sigmas_used: list[float]              # per-asset vol after B2 calibration
    asset_names: list[str]
    market_date: str
    iv_rmse_per_asset: list[float]        # B2 calibration quality
    ki_hit_prob: float
    expected_life_years: float


def load_deep_calib(model_path: str = "models/deep_calib.pt", device: str = "cpu"):
    """Load trained Deep Calibration model with normalization stats."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = DeepCalibNet(DeepCalibConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def calibrate_asset_to_heston(iv_surface_25: np.ndarray,
                                model, ckpt) -> tuple[HestonParams, float]:
    """Single asset's market IV surface → Heston params (B2 forward)."""
    p_fit, rmse = nn_calibrate(
        iv_surface_25, model,
        ckpt["x_mean"], ckpt["x_std"],
        ckpt["y_mean"], ckpt["y_std"],
        lr=5e-2, steps=300,
    )
    return p_fit, rmse


def els_daily_nav(
    product: StepDownELS,
    iv_surfaces_per_asset: list[np.ndarray],   # [asset0_25cells, asset1_25cells, ...]
    asset_names: list[str],
    r: float = 0.035,
    q: np.ndarray | None = None,
    corr: np.ndarray | None = None,
    market_date: str = "today",
    n_paths: int = 50_000,
    n_steps_per_year: int = 252,
    seed: int = 0,
    deep_calib_path: str = "models/deep_calib.pt",
    device: str = "cpu",
) -> ELSDailyNAVResult:
    """Full daily NAV pipeline.

    Steps:
      1. For each asset: B2 calibrate (IV surface → Heston params)
      2. Extract σ_today from sqrt(v0) per asset
      3. Layer A price_els() with calibrated σ
      4. Return fair value + diagnostics
    """
    n_assets = len(iv_surfaces_per_asset)
    if q is None:
        q = np.zeros(n_assets)
    if corr is None:
        corr = np.eye(n_assets)

    model, ckpt = load_deep_calib(deep_calib_path, device=device)

    # Per-asset Heston calibration
    sigmas = []
    rmses = []
    for iv in iv_surfaces_per_asset:
        p_fit, rmse = calibrate_asset_to_heston(iv, model, ckpt)
        # Spot vol from instantaneous variance
        sigmas.append(math.sqrt(max(p_fit.v0, 1e-6)))
        rmses.append(float(rmse))

    sigmas_arr = np.array(sigmas)

    # Layer A pricing
    res: ELSResult = price_els(
        product, r=r, q=q, sigma=sigmas_arr, corr=corr,
        n_paths=n_paths, n_steps_per_year=n_steps_per_year, seed=seed,
    )

    notional = float(product.notional)
    deviation_pct = (res.price - notional) / notional * 100

    return ELSDailyNAVResult(
        fair_value_krw=res.price,
        fair_value_stderr=res.stderr,
        issue_price_krw=notional,
        deviation_pct=deviation_pct,
        sigmas_used=[float(s) for s in sigmas],
        asset_names=asset_names,
        market_date=market_date,
        iv_rmse_per_asset=rmses,
        ki_hit_prob=res.ki_hit_prob,
        expected_life_years=res.expected_life,
    )


__all__ = ["ELSDailyNAVResult", "els_daily_nav",
           "load_deep_calib", "calibrate_asset_to_heston"]
