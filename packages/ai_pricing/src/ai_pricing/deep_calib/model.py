"""Deep Calibration NN: params → IV surface."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None  # type: ignore
    nn = None     # type: ignore


@dataclass
class DeepCalibConfig:
    in_dim: int = 5        # Heston params
    out_dim: int = 25      # surface grid points
    hidden: tuple[int, ...] = (256, 256, 256)
    activation: str = "elu"


class DeepCalibNet(nn.Module if _HAS_TORCH else object):
    def __init__(self, cfg: DeepCalibConfig | None = None):
        if not _HAS_TORCH:
            raise ImportError("pip install torch")
        super().__init__()
        cfg = cfg or DeepCalibConfig()
        self.cfg = cfg

        Act = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU}[cfg.activation]
        layers: list = []
        in_dim = cfg.in_dim
        for h in cfg.hidden:
            layers += [nn.Linear(in_dim, h), Act()]
            in_dim = h
        layers += [nn.Linear(in_dim, cfg.out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


__all__ = ["DeepCalibConfig", "DeepCalibNet"]
