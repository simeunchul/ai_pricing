"""Notebook W5 — Deep Calibration demo."""

# %%
# Prerequisite: python -m ai_pricing.deep_calib.train --n 2000 --epochs 40
import numpy as np
import time
from pathlib import Path

from pricing.heston import HestonParams
from ai_pricing.deep_calib.surface import iv_surface, STRIKES, MATURITIES

MODEL = "models/deep_calib.pt"
if not Path(MODEL).exists():
    print("Train first: python -m ai_pricing.deep_calib.train")
else:
    import torch
    from ai_pricing.deep_calib.model import DeepCalibNet, DeepCalibConfig
    from ai_pricing.deep_calib.calibrate import calibrate

    ckpt = torch.load(MODEL, map_location="cpu")
    model = DeepCalibNet(DeepCalibConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    true_p = HestonParams(kappa=1.5, theta=0.04, xi=0.4, rho=-0.6, v0=0.05)
    iv_target = iv_surface(true_p)

    t0 = time.perf_counter()
    p_fit, rmse = calibrate(iv_target, model,
                            ckpt["x_mean"], ckpt["x_std"],
                            ckpt["y_mean"], ckpt["y_std"],
                            lr=5e-2, steps=300)
    dt = time.perf_counter() - t0
    print(f"True:   kappa={true_p.kappa:.3f} theta={true_p.theta:.4f} xi={true_p.xi:.3f} rho={true_p.rho:+.3f} v0={true_p.v0:.4f}")
    print(f"Fit:    kappa={p_fit.kappa:.3f} theta={p_fit.theta:.4f} xi={p_fit.xi:.3f} rho={p_fit.rho:+.3f} v0={p_fit.v0:.4f}")
    print(f"IV RMSE: {rmse*100:.3f} vol pts in {dt*1000:.0f}ms")
