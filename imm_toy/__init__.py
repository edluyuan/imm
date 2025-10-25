"""Toy implementation of Inductive Moment Matching for 2D point clouds."""

from .datasets import sample_mixture_of_gaussians
from .models import Generator, MomentNetwork
from .trainer import IMMTrainer, TrainingConfig

__all__ = [
    "sample_mixture_of_gaussians",
    "Generator",
    "MomentNetwork",
    "IMMTrainer",
    "TrainingConfig",
]
