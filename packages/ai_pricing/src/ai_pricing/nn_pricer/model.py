"""MLP pricer model.

Input features (5):
    [log(S/K), T, r, q, sigma]

Output: undiscounted call price scaled by K, i.e. model(x) ≈ C / K in [0, S/K].
Using log-moneyness and normalized output keeps gradients well-conditioned.
"""

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
class NNPricerConfig:
    hidden: tuple[int, ...] = (128, 128, 64)
    activation: str = "tanh"  # "tanh" | "relu" | "silu"
    output_softplus: bool = True  # enforce price > 0


def _act(name: str):
    if not _HAS_TORCH:
        raise ImportError("Install torch: pip install torch")
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[name]


class NNPricer(nn.Module if _HAS_TORCH else object):
    """MLP: [logM, T, r, q, sigma] → C/K."""

    def __init__(self, cfg: NNPricerConfig | None = None):
        if not _HAS_TORCH:
            raise ImportError("Install torch: pip install torch")
        super().__init__()
        cfg = cfg or NNPricerConfig()
        self.cfg = cfg

        Act = _act(cfg.activation)
        layers: list = []
        in_dim = 5
        for h in cfg.hidden:
            layers += [nn.Linear(in_dim, h), Act()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        if cfg.output_softplus:
            layers += [nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (N, 5)
        return self.net(x).squeeze(-1)


__all__ = ["NNPricer", "NNPricerConfig"]
