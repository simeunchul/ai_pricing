"""Real-time calibration: given market IV surface, solve for Heston params via grad desc
on the pretrained surrogate network."""

from __future__ import annotations

import numpy as np

from pricing.heston import HestonParams
from ai_pricing.deep_calib.sampler import PARAM_RANGES


def _clamp(x, lo, hi):
    import torch
    return torch.max(torch.min(x, torch.as_tensor(hi, dtype=x.dtype)),
                     torch.as_tensor(lo, dtype=x.dtype))


def calibrate(
    iv_market: np.ndarray,
    model,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    lr: float = 5e-2,
    steps: int = 300,
    init: np.ndarray | None = None,
    device: str = "cpu",
) -> tuple[HestonParams, float]:
    """Gradient descent on Heston param vector using pretrained `model`.

    Returns (HestonParams, final_rmse_in_vol_points).
    """
    import torch

    lo = np.array([PARAM_RANGES[k][0] for k in ("kappa", "theta", "xi", "rho", "v0")])
    hi = np.array([PARAM_RANGES[k][1] for k in ("kappa", "theta", "xi", "rho", "v0")])

    if init is None:
        init = (lo + hi) / 2.0

    theta = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
    xm = torch.from_numpy(x_mean.astype(np.float32)).to(device)
    xs = torch.from_numpy(x_std.astype(np.float32)).to(device)
    ym = torch.from_numpy(y_mean.astype(np.float32)).to(device)
    ys = torch.from_numpy(y_std.astype(np.float32)).to(device)
    target = torch.from_numpy(iv_market.astype(np.float32)).to(device)

    opt = torch.optim.Adam([theta], lr=lr)
    lo_t = torch.from_numpy(lo.astype(np.float32)).to(device)
    hi_t = torch.from_numpy(hi.astype(np.float32)).to(device)

    for _ in range(steps):
        x_norm = (theta - xm) / xs
        pred_norm = model(x_norm.unsqueeze(0)).squeeze(0)
        pred = pred_norm * ys + ym
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # project to feasible box
        with torch.no_grad():
            theta.data = torch.max(torch.min(theta.data, hi_t), lo_t)

    final = theta.detach().cpu().numpy()
    p = HestonParams(kappa=float(final[0]), theta=float(final[1]),
                     xi=float(final[2]), rho=float(final[3]), v0=float(final[4]))
    rmse = float(np.sqrt(loss.item()))
    return p, rmse


__all__ = ["calibrate"]
