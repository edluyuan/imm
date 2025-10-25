"""Utility functions for constructing toy 2D point cloud datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


@dataclass
class GaussianComponent:
    """Parameters describing a 2D Gaussian component."""

    mean: torch.Tensor
    cov: torch.Tensor
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mean.shape != (2,):
            raise ValueError("GaussianComponent.mean must be shape (2,)")
        if self.cov.shape != (2, 2):
            raise ValueError("GaussianComponent.cov must be shape (2, 2)")
        if self.weight <= 0:
            raise ValueError("GaussianComponent.weight must be positive")


def sample_mixture_of_gaussians(
    num_samples: int,
    components: Iterable[GaussianComponent],
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from a mixture of 2D Gaussians.

    Parameters
    ----------
    num_samples:
        Number of points to sample.
    components:
        Iterable of :class:`GaussianComponent` objects describing the mixture.
    device:
        Optional device for the returned tensor.
    generator:
        Optional torch random generator for deterministic sampling.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(num_samples, 2)`` containing the sampled points.
    """

    mixture = tuple(components)
    if not mixture:
        raise ValueError("At least one GaussianComponent is required")

    weights = torch.tensor([c.weight for c in mixture], dtype=torch.float32)
    weights = weights / weights.sum()
    categorical = torch.distributions.Categorical(weights)

    choices = categorical.sample((num_samples,), generator=generator)
    samples = []
    for idx in choices:
        comp = mixture[int(idx)]
        mvn = torch.distributions.MultivariateNormal(
            comp.mean.to(device=device), comp.cov.to(device=device)
        )
        samples.append(mvn.sample(generator=generator))

    return torch.stack(samples, dim=0)


def default_mixture(device: torch.device | None = None) -> Tuple[GaussianComponent, ...]:
    """Return a simple mixture used in the examples.

    The configuration mirrors the clusters used in the IMM paper for
    illustrative 2D experiments: four well-separated Gaussian blobs.
    """

    means = torch.tensor(
        [
            (-2.0, -2.0),
            (-2.0, 2.0),
            (2.0, -2.0),
            (2.0, 2.0),
        ],
        dtype=torch.float32,
        device=device,
    )
    cov = torch.eye(2, dtype=torch.float32, device=device) * 0.25
    return tuple(
        GaussianComponent(mean=means[i], cov=cov.clone(), weight=1.0) for i in range(4)
    )


__all__ = ["GaussianComponent", "sample_mixture_of_gaussians", "default_mixture"]
