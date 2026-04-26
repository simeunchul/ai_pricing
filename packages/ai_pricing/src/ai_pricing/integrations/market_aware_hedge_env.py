"""B2 → B4 hedge env σ integration.

B4 의 학습 / 평가 환경의 σ 를 hardcode (보통 0.20) 가 아니라
B2 가 매일 시장 IV 에서 calibrate 한 vol 로 동적 사용.

Training pattern:
  morning: B2 calibrate today's σ → BuehlerConfig(sigma=σ_today) → train B4
  실무에선 매일 새 정책으로 train 하지 않고 weekly retrain. 하지만 evaluate 환경은
  매일 σ 갱신.

Evaluation pattern:
  policy 는 어제 학습 (σ_train).
  오늘 σ_today 환경에서 평가 → 강건성 측정 (σ shift robustness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ai_pricing.deep_calib.model import DeepCalibNet, DeepCalibConfig
from ai_pricing.deep_calib.calibrate import calibrate as nn_calibrate


@dataclass
class MarketCalibratedSigma:
    sigma_calibrated: float
    sigma_baseline: float          # default 0.20
    sigma_shift: float
    iv_rmse_vp: float
    market_date: str


def get_market_sigma(
    iv_surface_25: np.ndarray,
    market_date: str = "today",
    deep_calib_path: str = "models/deep_calib.pt",
    sigma_baseline: float = 0.20,
    device: str = "cpu",
) -> MarketCalibratedSigma:
    """B2 forward: 시장 IV → Heston params → spot σ."""
    ckpt = torch.load(deep_calib_path, map_location=device, weights_only=False)
    model = DeepCalibNet(DeepCalibConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    p_fit, rmse = nn_calibrate(
        iv_surface_25, model,
        ckpt["x_mean"], ckpt["x_std"],
        ckpt["y_mean"], ckpt["y_std"],
        lr=5e-2, steps=300,
    )
    sigma_today = math.sqrt(max(p_fit.v0, 1e-6))

    return MarketCalibratedSigma(
        sigma_calibrated=sigma_today,
        sigma_baseline=sigma_baseline,
        sigma_shift=sigma_today - sigma_baseline,
        iv_rmse_vp=float(rmse * 100),
        market_date=market_date,
    )


def buehler_cfg_with_market_sigma(market_sigma: MarketCalibratedSigma,
                                    base_kwargs: dict | None = None):
    """Returns BuehlerConfig with sigma=market_sigma.sigma_calibrated."""
    from ai_hedging.agents.buehler_pg import BuehlerConfig
    base_kwargs = base_kwargs or {}
    base_kwargs.pop("sigma", None)
    return BuehlerConfig(sigma=market_sigma.sigma_calibrated, **base_kwargs)


__all__ = ["MarketCalibratedSigma", "get_market_sigma", "buehler_cfg_with_market_sigma"]
