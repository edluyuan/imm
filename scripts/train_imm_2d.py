"""Train the toy IMM model on a synthetic 2D point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch

from imm_toy import IMMTrainer, TrainingConfig, default_mixture, sample_mixture_of_gaussians


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where plots and samples will be written",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def make_dataset_sampler(device: torch.device) -> Callable[[int], torch.Tensor]:
    mixture = default_mixture(device=device)

    def _sample(batch_size: int) -> torch.Tensor:
        return sample_mixture_of_gaussians(batch_size, mixture, device=device)

    return _sample


def main() -> None:
    args = make_argparser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    sampler = make_dataset_sampler(device)
    config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        log_every=max(args.steps // 10, 1),
    )
    trainer = IMMTrainer(config, sampler)

    history = trainer.train()
    samples = trainer.sample(4096)
    real = sampler(4096).cpu()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(real[:, 0], real[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Real samples")
    axes[0].set_aspect("equal")

    axes[1].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color="tab:orange")
    axes[1].set_title("IMM generated samples")
    axes[1].set_aspect("equal")

    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plot_path = args.output_dir / "imm_2d_point_cloud.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    log_path = args.output_dir / "training_log.pt"
    torch.save(history, log_path)

    print(f"Saved plot to {plot_path}")
    print(f"Saved metrics to {log_path}")


if __name__ == "__main__":
    main()
