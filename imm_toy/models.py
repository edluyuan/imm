"""Model components used in the toy IMM experiment."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class Generator(nn.Module):
    """Simple MLP generator mapping latent noise to 2D points."""

    def __init__(self, latent_dim: int = 4, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(z)


class MomentNetwork(nn.Module):
    """Feature extractor that learns informative moments."""

    def __init__(self, hidden_dim: int = 128, num_features: int = 8, num_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = 2
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_features))
        self.net = nn.Sequential(*layers)
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Apply orthogonal initialization to Linear layers."""

    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


__all__ = ["Generator", "MomentNetwork", "orthogonal_init"]
