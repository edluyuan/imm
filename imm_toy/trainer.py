"""Training loop for the toy IMM experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn, optim

from .models import Generator, MomentNetwork, orthogonal_init


@dataclass
class TrainingConfig:
    """Configuration controlling IMM training."""

    latent_dim: int = 4
    generator_hidden: int = 128
    generator_layers: int = 3
    moment_hidden: int = 128
    moment_layers: int = 4
    moment_features: int = 16
    batch_size: int = 256
    steps: int = 5000
    lr_generator: float = 1e-3
    lr_moment: float = 1e-3
    moment_reg: float = 1e-1
    device: str = "cpu"
    log_every: int = 200


class IMMTrainer:
    """Small trainer implementing the IMM min-max objective."""

    def __init__(
        self,
        config: TrainingConfig,
        dataset_sampler: Callable[[int], torch.Tensor],
        *,
        feature_momentum: float = 0.99,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.generator = Generator(
            latent_dim=config.latent_dim,
            hidden_dim=config.generator_hidden,
            num_layers=config.generator_layers,
        ).to(self.device)
        self.moment = MomentNetwork(
            hidden_dim=config.moment_hidden,
            num_features=config.moment_features,
            num_layers=config.moment_layers,
        ).to(self.device)
        self.generator.apply(orthogonal_init)
        self.moment.apply(orthogonal_init)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=config.lr_generator)
        self.optim_m = optim.Adam(self.moment.parameters(), lr=config.lr_moment)

        self.dataset_sampler = dataset_sampler
        self.feature_momentum = feature_momentum
        self.register_buffer("running_real", torch.zeros(config.moment_features))
        self.register_buffer("running_fake", torch.zeros(config.moment_features))

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor.to(self.device))

    def sample_real(self, batch_size: int) -> torch.Tensor:
        return self.dataset_sampler(batch_size).to(self.device)

    def sample_fake(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        return self.generator(z)

    def compute_moment_loss(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_feats = self.moment(x_real)
        fake_feats = self.moment(x_fake)
        diff = real_feats.mean(0) - fake_feats.mean(0)
        loss = (diff**2).sum()

        running_real = (
            self.feature_momentum * self.running_real
            + (1 - self.feature_momentum) * real_feats.mean(0).detach()
        )
        running_fake = (
            self.feature_momentum * self.running_fake
            + (1 - self.feature_momentum) * fake_feats.mean(0).detach()
        )
        self.running_real = running_real
        self.running_fake = running_fake

        reg = real_feats.pow(2).mean() + fake_feats.pow(2).mean()
        return loss, reg

    def train(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for step in range(1, self.config.steps + 1):
            real = self.sample_real(self.config.batch_size)
            fake = self.sample_fake(self.config.batch_size)

            loss, reg = self.compute_moment_loss(real, fake)

            # Update generator (minimize feature difference)
            self.optim_g.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=10.0)
            self.optim_g.step()

            # Update moment network (maximize difference, regulate magnitude)
            self.optim_m.zero_grad()
            moment_loss = -loss + self.config.moment_reg * reg
            moment_loss.backward()
            nn.utils.clip_grad_norm_(self.moment.parameters(), max_norm=10.0)
            self.optim_m.step()

            if step % self.config.log_every == 0 or step == self.config.steps:
                with torch.no_grad():
                    wasserstein = torch.linalg.vector_norm(self.running_real - self.running_fake).item()
                    history.append(
                        {
                            "step": step,
                            "moment_loss": loss.item(),
                            "regularizer": reg.item(),
                            "running_gap": wasserstein,
                        }
                    )
        return history

    def sample(self, num_samples: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        z = torch.randn(num_samples, self.config.latent_dim, device=self.device, generator=generator)
        with torch.no_grad():
            samples = self.generator(z)
        return samples.cpu()


__all__ = ["IMMTrainer", "TrainingConfig"]
