# Inductive Moment Matching Toy Implementation

This repository provides a compact, well-documented implementation of the toy
2D point-cloud experiment from the paper *Inductive Moment Matching* by Li et al.
(2025). The goal is to reproduce the adversarial moment matching scheme in a
minimal setting so that the core intuition of IMM is easy to understand and
extend.

## Getting Started

Create a virtual environment, install dependencies, and run the training script:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_imm_2d.py --steps 2000 --batch-size 512
```

The script trains a small generator against a learnable moment network on a
mixture of four Gaussians. After training it writes a scatter plot comparing the
real and generated point clouds to `outputs/imm_2d_point_cloud.png` and saves the
training log to `outputs/training_log.pt`.

## Project Structure

- `imm_toy/`
  - `datasets.py` — helpers for sampling synthetic 2D point clouds.
  - `models.py` — generator and moment feature networks.
  - `trainer.py` — IMM min-max training loop.
- `scripts/train_imm_2d.py` — command-line entry point for running the toy
  experiment and exporting visualizations.

The code is designed to closely mirror the training dynamics described in the
original paper while remaining easy to modify for further experimentation.
